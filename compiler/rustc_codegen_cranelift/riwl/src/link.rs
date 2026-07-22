//! The link driver: load → resolve (lazy archive fixpoint) → layout → apply
//! relocations in place → emit.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{bail, ensure, Context, Result};

use crate::archive::{is_wasm, parse_archive};
use crate::emit::{valtype_byte, write_name, Emitter};
use crate::object::{FuncType, InputObject, SymDef, SymKind, Symbol};
use crate::{
    patch_le32, patch_sleb5, patch_uleb5, sleb, uleb, R_WASM_FUNCTION_INDEX_LEB,
    R_WASM_GLOBAL_INDEX_LEB, R_WASM_MEMORY_ADDR_I32, R_WASM_MEMORY_ADDR_LEB,
    R_WASM_MEMORY_ADDR_REL_SLEB, R_WASM_MEMORY_ADDR_SLEB, R_WASM_TABLE_INDEX_I32,
    R_WASM_TABLE_INDEX_SLEB, R_WASM_TABLE_NUMBER_LEB, R_WASM_TYPE_INDEX_LEB,
};

pub struct LinkOptions {
    /// Objects and archives, in command-line order.
    pub inputs: Vec<PathBuf>,
    /// `-l` names, searched in `lib_paths` as lib<name>.a.
    pub libs: Vec<String>,
    pub lib_paths: Vec<PathBuf>,
    /// `--export` names (also root the resolution).
    pub exports: Vec<String>,
    pub no_entry: bool,
    pub stack_size: u64,
    pub stack_first: bool,
    /// lld's --allow-undefined: unresolved undefined FUNCTIONS become env
    /// imports instead of hard errors.
    pub allow_undefined: bool,
}

impl Default for LinkOptions {
    fn default() -> Self {
        LinkOptions {
            inputs: vec![],
            libs: vec![],
            lib_paths: vec![],
            exports: vec![],
            no_entry: false,
            stack_size: 1 << 20,
            stack_first: true,
            allow_undefined: false,
        }
    }
}

/// Where a resolved symbol points.
#[derive(Copy, Clone, Debug)]
enum Target {
    /// (included-object position, LOCAL wasm function index in that object).
    Func(usize, u32),
    /// Runtime import ordinal (wasi etc.).
    FuncImport(u32),
    /// Synthetic trap stub for an unresolved weak function, by type.
    FuncStub(u32),
    /// The synthesized __wasm_call_ctors.
    FuncCtors,
    /// (included-object position, segment index, offset within segment).
    Data(usize, u32, u32),
    /// Absolute address (linker-provided data symbols, weak-undef data).
    DataAbs(u32),
    /// Output global index.
    Global(u32),
    Table,
}

struct Linked {
    objects: Vec<InputObject>,
}

pub fn link(opts: &LinkOptions) -> Result<Vec<u8>> {
    // Phase timing (RIWL_TIMINGS=1): one machine-readable stderr line at the end.
    let timings_on = std::env::var_os("RIWL_TIMINGS").is_some_and(|v| v == "1");
    let t_start = std::time::Instant::now();
    let mut t_last = t_start;
    let mut phase_ms: Vec<(&str, u128)> = Vec::new();
    let mut mark = |name: &'static str, t_last: &mut std::time::Instant| {
        phase_ms.push((name, t_last.elapsed().as_millis()));
        *t_last = std::time::Instant::now();
    };

    // ---- load ----
    let mut included: Vec<InputObject> = Vec::new();
    let mut lazy: Vec<(String, InputObject)> = Vec::new(); // (archive display, obj)

    let mut load_archive = |path: &std::path::Path, lazy: &mut Vec<(String, InputObject)>| -> Result<()> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let disp = path.display().to_string();
        for m in parse_archive(&disp, &bytes)? {
            let data = &bytes[m.range.0..m.range.1];
            if is_wasm(data) {
                let obj = InputObject::parse(&format!("{disp}({})", m.name), data)?;
                lazy.push((disp.clone(), obj));
            }
        }
        Ok(())
    };

    for path in &opts.inputs {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        if is_wasm(&bytes) {
            included.push(InputObject::parse(&path.display().to_string(), &bytes)?);
        } else {
            load_archive(path, &mut lazy)?;
        }
    }
    for libname in &opts.libs {
        let fname = format!("lib{libname}.a");
        let found = opts
            .lib_paths
            .iter()
            .map(|d| d.join(&fname))
            .find(|p| p.exists())
            .with_context(|| format!("library not found: -l{libname}"))?;
        load_archive(&found, &mut lazy)?;
    }

    mark("load", &mut t_last);

    // ---- resolve ----
    // Global (non-local) symbol map: name -> (included pos, sym idx).
    let mut global: HashMap<String, (usize, usize)> = HashMap::new();
    // Lazy index: name -> first lazy member defining it.
    let mut lazy_index: HashMap<String, usize> = HashMap::new();
    for (li, (_, obj)) in lazy.iter().enumerate() {
        for s in &obj.symbols {
            if s.def.is_some() && !s.is_local() {
                lazy_index.entry(s.name.clone()).or_insert(li);
            }
        }
    }

    fn add_object_symbols(
        global: &mut HashMap<String, (usize, usize)>,
        objects: &[InputObject],
        pos: usize,
    ) -> Result<()> {
        let obj = &objects[pos];
        for (si, s) in obj.symbols.iter().enumerate() {
            if s.is_local() || s.def.is_none() || s.name.is_empty() {
                continue;
            }
            match global.entry(s.name.clone()) {
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert((pos, si));
                }
                std::collections::hash_map::Entry::Occupied(mut o) => {
                    let (epos, esi) = *o.get();
                    let existing = &objects[epos].symbols[esi];
                    match (existing.is_weak(), s.is_weak()) {
                        (true, false) => {
                            o.insert((pos, si));
                        }
                        (false, false) => bail!(
                            "duplicate symbol `{}` in {} and {}",
                            s.name,
                            objects[epos].name,
                            obj.name
                        ),
                        // first weak wins; strong already present wins
                        _ => {}
                    }
                }
            }
        }
        Ok(())
    }

    for pos in 0..included.len() {
        add_object_symbols(&mut global, &included, pos)?;
    }

    // Fixpoint: pull lazy members that define needed undefined symbols.
    let mut pulled: Vec<bool> = vec![false; lazy.len()];
    loop {
        let mut want: Option<usize> = None;
        'scan: for obj in &included {
            for s in &obj.symbols {
                if s.is_undefined()
                    && !s.name.is_empty()
                    && !global.contains_key(&s.name)
                {
                    if let Some(&li) = lazy_index.get(&s.name) {
                        if !pulled[li] {
                            want = Some(li);
                            break 'scan;
                        }
                    }
                }
            }
        }
        let Some(li) = want else { break };
        pulled[li] = true;
        let (_arch, obj) = &lazy[li];
        included.push(clone_object(obj));
        let pos = included.len() - 1;
        add_object_symbols(&mut global, &included, pos)?;
    }

    mark("resolve", &mut t_last);

    // ---- classify remaining undefined ----
    // Output-level runtime imports (module, field, FuncType) — dedup by (m,f).
    let mut runtime_imports: Vec<(String, String, FuncType)> = Vec::new();
    let mut runtime_import_ix: HashMap<(String, String), u32> = HashMap::new();
    // Trap stubs for unresolved weak funcs, by type.
    let mut stub_types: Vec<FuncType> = Vec::new();
    let mut stub_ix: HashMap<FuncType, u32> = HashMap::new();
    // name -> special resolution
    let mut special: HashMap<String, Target> = HashMap::new();
    // (init value, mutable) for synthetic globals after __stack_pointer.
    let mut synth_globals: Vec<(u32, bool)> = Vec::new();
    let mut need_ctors = false;

    // Linker-provided data symbols get their values after layout; collect names.
    const ABS_SYMS: &[&str] = &[
        "__heap_base",
        "__heap_end",
        "__data_end",
        "__global_base",
        "__dso_handle",
        "__wasm_first_page_end",
    ];

    let mut missing: Vec<String> = Vec::new();
    for obj in &included {
        for s in &obj.symbols {
            if !s.is_undefined() || s.name.is_empty() || global.contains_key(&s.name) {
                continue;
            }
            if special.contains_key(&s.name) {
                continue;
            }
            match (s.kind, s.name.as_str()) {
                (SymKind::Global, "__stack_pointer") => {
                    special.insert(s.name.clone(), Target::Global(0));
                }
                // Static-link values for the linker-provided globals (wasm-ld
                // parity): memory/table bases are constants; __tls_base stays 0 -
                // single-threaded TLS accesses go through plain MEMORY_ADDR
                // relocs (no TLS reloc types exist in our input universe).
                (
                    SymKind::Global,
                    n @ ("__memory_base" | "__table_base" | "__tls_base" | "__tls_size"
                    | "__tls_align"),
                ) => {
                    let (value, mutable) = match n {
                        "__table_base" => (1u32, false),
                        "__tls_base" => (0, true),
                        "__tls_align" => (1, false),
                        _ => (0, false),
                    };
                    let idx = 1 + synth_globals.len() as u32;
                    synth_globals.push((value, mutable));
                    special.insert(s.name.clone(), Target::Global(idx));
                }
                (SymKind::Table, "__indirect_function_table") => {
                    special.insert(s.name.clone(), Target::Table);
                }
                (SymKind::Func, "__wasm_call_ctors") => {
                    need_ctors = true;
                    special.insert(s.name.clone(), Target::FuncCtors);
                }
                (SymKind::Data, n) if ABS_SYMS.contains(&n) => {
                    special.insert(s.name.clone(), Target::DataAbs(0)); // patched after layout
                }
                (SymKind::Func, _) => {
                    let (m, f) = s.import.clone().context("undef func without import")?;
                    if m != "env" || opts.allow_undefined {
                        // Runtime (WASI) import.
                        let key = (m.clone(), f.clone());
                        if !runtime_import_ix.contains_key(&key) {
                            runtime_import_ix.insert(key, runtime_imports.len() as u32);
                            runtime_imports.push((
                                m,
                                f,
                                s.func_type.clone().context("import type")?,
                            ));
                        }
                        special.insert(
                            s.name.clone(),
                            Target::FuncImport(runtime_import_ix[&(
                                s.import.clone().unwrap().0,
                                s.import.clone().unwrap().1,
                            )]),
                        );
                    } else if s.is_weak() {
                        let ft = s.func_type.clone().context("weak undef type")?;
                        let ix = *stub_ix.entry(ft.clone()).or_insert_with(|| {
                            stub_types.push(ft.clone());
                            (stub_types.len() - 1) as u32
                        });
                        special.insert(s.name.clone(), Target::FuncStub(ix));
                    } else {
                        missing.push(format!("{} (referenced by {})", s.name, obj.name));
                    }
                }
                (SymKind::Data, _) if s.is_weak() => {
                    special.insert(s.name.clone(), Target::DataAbs(0));
                }
                _ => missing.push(format!("{} (referenced by {})", s.name, obj.name)),
            }
        }
    }
    if !missing.is_empty() {
        missing.sort();
        missing.dedup();
        bail!("undefined symbols:\n  {}", missing.join("\n  "));
    }

    // ---- layout: types ----
    let mut out_types: Vec<FuncType> = Vec::new();
    let mut type_ix: HashMap<FuncType, u32> = HashMap::new();
    let mut intern_type = |t: &FuncType, out_types: &mut Vec<FuncType>, type_ix: &mut HashMap<FuncType, u32>| -> u32 {
        *type_ix.entry(t.clone()).or_insert_with(|| {
            out_types.push(t.clone());
            (out_types.len() - 1) as u32
        })
    };
    // per-object local type index -> output type index
    let mut obj_type_map: Vec<Vec<u32>> = Vec::with_capacity(included.len());
    for obj in &included {
        obj_type_map.push(
            obj.types.iter().map(|t| intern_type(t, &mut out_types, &mut type_ix)).collect(),
        );
    }
    let rt_import_types: Vec<u32> = runtime_imports
        .iter()
        .map(|(_, _, t)| intern_type(t, &mut out_types, &mut type_ix))
        .collect();
    let stub_out_types: Vec<u32> =
        stub_types.iter().map(|t| intern_type(t, &mut out_types, &mut type_ix)).collect();
    let void_ty = intern_type(
        &FuncType { params: vec![], results: vec![] },
        &mut out_types,
        &mut type_ix,
    );

    // ---- layout: function index space ----
    // imports | per-object defined funcs | stubs | ctors
    let n_imports = runtime_imports.len() as u32;
    let mut obj_func_base: Vec<u32> = Vec::with_capacity(included.len());
    let mut next = n_imports;
    for obj in &included {
        obj_func_base.push(next);
        next += obj.func_type_idx.len() as u32;
    }
    let stubs_base = next;
    next += stub_types.len() as u32;
    let ctors_index = next;
    if need_ctors || opts.exports.iter().any(|e| e == "__wasm_call_ctors") {
        next += 1;
    }
    let _total_funcs = next;

    // ---- layout: data ----
    let page = 0x1_0000u64;
    let mut cursor: u64 = if opts.stack_first { opts.stack_size } else { 1024 };
    let mut seg_addrs: Vec<Vec<u32>> = Vec::with_capacity(included.len());
    for obj in &included {
        let mut addrs = Vec::with_capacity(obj.segments.len());
        for seg in &obj.segments {
            let align = 1u64 << seg.align_log2;
            cursor = (cursor + align - 1) & !(align - 1);
            addrs.push(cursor as u32);
            cursor += seg.len as u64;
        }
        seg_addrs.push(addrs);
    }
    let data_end = cursor as u32;
    let heap_base = (data_end + 15) & !15;
    let (sp_init, mem_min) = if opts.stack_first {
        (opts.stack_size as u32, (heap_base as u64).div_ceil(page))
    } else {
        let sp = ((data_end as u64 + opts.stack_size + 15) & !15) as u32;
        (sp, (sp as u64).div_ceil(page))
    };
    let heap_base = if opts.stack_first { heap_base } else { (sp_init + 15) & !15 };

    let abs_values: HashMap<&str, u32> = HashMap::from([
        ("__heap_base", heap_base),
        ("__heap_end", (mem_min * page) as u32),
        ("__data_end", data_end),
        ("__global_base", if opts.stack_first { opts.stack_size as u32 } else { 1024 }),
        ("__dso_handle", 0),
        ("__wasm_first_page_end", 0x1_0000),
    ]);

    // ---- resolution helper ----
    let resolve = |obj_pos: usize, sym_idx: u32| -> Result<Target> {
        let obj = &included[obj_pos];
        let s: &Symbol = obj
            .symbols
            .get(sym_idx as usize)
            .with_context(|| format!("{}: symbol {sym_idx} out of range", obj.name))?;
        let resolve_def = |dpos: usize, dsym: &Symbol| -> Result<Target> {
            let dobj = &included[dpos];
            Ok(match dsym.def.as_ref().unwrap() {
                SymDef::Func { local_index } => {
                    ensure!(
                        *local_index >= dobj.func_import_count(),
                        "{}: defined func symbol `{}` points at an import",
                        dobj.name,
                        dsym.name
                    );
                    Target::Func(dpos, *local_index)
                }
                SymDef::Data { segment, offset, .. } => Target::Data(dpos, *segment, *offset),
                SymDef::Global { .. } => Target::Global(0),
                SymDef::Table { .. } => Target::Table,
            })
        };
        if s.def.is_some() && (s.is_local() || !global.contains_key(&s.name)) {
            return resolve_def(obj_pos, s);
        }
        if let Some(&(dpos, dsi)) = global.get(&s.name) {
            let dsym = &included[dpos].symbols[dsi];
            // Signature tripwire: undefined func resolved to a definition with a
            // different type is the ABI bug class — fail loudly like wasm-ld.
            if s.def.is_none() && s.kind == SymKind::Func {
                if let (Some(a), Some(b)) = (&s.func_type, &dsym.func_type) {
                    ensure!(
                        a == b,
                        "function signature mismatch for `{}`:\n  wanted {:?} ({})\n  defined {:?} ({})",
                        s.name,
                        a,
                        obj.name,
                        b,
                        included[dpos].name
                    );
                }
            }
            return resolve_def(dpos, dsym);
        }
        if let Some(t) = special.get(&s.name) {
            if let Target::DataAbs(_) = t {
                let v = abs_values.get(s.name.as_str()).copied().unwrap_or(0);
                return Ok(Target::DataAbs(v));
            }
            return Ok(*t);
        }
        bail!("{}: unresolved symbol `{}`", obj.name, s.name)
    };

    let out_func_index = |t: &Target| -> Result<u32> {
        Ok(match t {
            Target::Func(pos, local) => {
                obj_func_base[*pos] + (local - included[*pos].func_import_count())
            }
            Target::FuncImport(i) => *i,
            Target::FuncStub(i) => stubs_base + i,
            Target::FuncCtors => ctors_index,
            other => bail!("expected function target, got {other:?}"),
        })
    };
    let data_addr = |t: &Target| -> Result<u32> {
        Ok(match t {
            Target::Data(pos, seg, off) => seg_addrs[*pos][*seg as usize] + off,
            Target::DataAbs(v) => *v,
            other => bail!("expected data target, got {other:?}"),
        })
    };

    // ---- table slots (address-taken functions) ----
    let mut table_slots: Vec<u32> = Vec::new(); // output func index per slot (slot i = index i+1)
    let mut slot_of: HashMap<u32, u32> = HashMap::new();
    for pos in 0..included.len() {
        let relocs: Vec<crate::object::RelocEntry> = included[pos]
            .code_relocs
            .iter()
            .chain(included[pos].data_relocs.iter())
            .copied()
            .collect();
        for r in relocs {
            if r.ty == R_WASM_TABLE_INDEX_SLEB || r.ty == R_WASM_TABLE_INDEX_I32 {
                let t = resolve(pos, r.index)?;
                let fi = out_func_index(&t)?;
                slot_of.entry(fi).or_insert_with(|| {
                    table_slots.push(fi);
                    table_slots.len() as u32 // slot 0 is null; first entry at 1
                });
            }
        }
    }

    mark("layout", &mut t_last);

    // ---- compute all patches (immutable pass), then apply in place ----
    let mut code_patches: Vec<Vec<(usize, PatchVal)>> = vec![Vec::new(); included.len()];
    let mut data_patches: Vec<Vec<(usize, PatchVal)>> = vec![Vec::new(); included.len()];
    for pos in 0..included.len() {
        let code_relocs = included[pos].code_relocs.clone();
        let data_relocs = included[pos].data_relocs.clone();
        for r in &code_relocs {
            let off = r.offset as usize;
            let val: PatchVal = match r.ty {
                R_WASM_FUNCTION_INDEX_LEB => {
                    PatchVal::U(out_func_index(&resolve(pos, r.index)?)?)
                }
                R_WASM_TYPE_INDEX_LEB => PatchVal::U(
                    *obj_type_map[pos]
                        .get(r.index as usize)
                        .with_context(|| format!("{}: type reloc", included[pos].name))?,
                ),
                R_WASM_GLOBAL_INDEX_LEB => match resolve(pos, r.index)? {
                    Target::Global(g) => PatchVal::U(g),
                    other => bail!("global reloc resolved to {other:?}"),
                },
                R_WASM_TABLE_NUMBER_LEB => PatchVal::U(0),
                R_WASM_TABLE_INDEX_SLEB => {
                    let fi = out_func_index(&resolve(pos, r.index)?)?;
                    PatchVal::S(slot_of[&fi] as i32)
                }
                R_WASM_MEMORY_ADDR_LEB => {
                    PatchVal::U((data_addr(&resolve(pos, r.index)?)? as i64 + r.addend) as u32)
                }
                R_WASM_MEMORY_ADDR_SLEB => {
                    PatchVal::S((data_addr(&resolve(pos, r.index)?)? as i64 + r.addend) as i32)
                }
                R_WASM_MEMORY_ADDR_I32 => {
                    PatchVal::I32((data_addr(&resolve(pos, r.index)?)? as i64 + r.addend) as u32)
                }
                R_WASM_TABLE_INDEX_I32 => {
                    let fi = out_func_index(&resolve(pos, r.index)?)?;
                    PatchVal::I32(slot_of[&fi])
                }
                // Relative to __memory_base, which is 0 in a static link — so
                // identical to MEMORY_ADDR_SLEB. (crt1-command.o carries these.)
                R_WASM_MEMORY_ADDR_REL_SLEB => {
                    PatchVal::S((data_addr(&resolve(pos, r.index)?)? as i64 + r.addend) as i32)
                }
                other => bail!("{}: unsupported code reloc type {other}", included[pos].name),
            };
            code_patches[pos].push((off, val));
        }
        for r in &data_relocs {
            let off = r.offset as usize;
            let val = match r.ty {
                R_WASM_MEMORY_ADDR_I32 => {
                    PatchVal::I32((data_addr(&resolve(pos, r.index)?)? as i64 + r.addend) as u32)
                }
                R_WASM_TABLE_INDEX_I32 => {
                    let fi = out_func_index(&resolve(pos, r.index)?)?;
                    PatchVal::I32(slot_of[&fi])
                }
                other => bail!("{}: unsupported data reloc type {other}", included[pos].name),
            };
            data_patches[pos].push((off, val));
        }
    }
    // ---- ctors resolution must also run before the mutable pass ----
    let mut ctors_calls: Vec<(u32, u32)> = Vec::new(); // (priority, out fn idx)
    for pos in 0..included.len() {
        for &(prio, sym) in &included[pos].init_funcs.clone() {
            let fi = out_func_index(&resolve(pos, sym)?)?;
            ctors_calls.push((prio, fi));
        }
    }
    ctors_calls.sort();
    // ---- entry/export resolution (still inside the immutable-borrow region) ----
    let entry_name = "_start";
    let mut export_list: Vec<(String, u8, u32)> = vec![("memory".into(), 0x02, 0)];
    {
        let mut exported = std::collections::HashSet::new();
        if !opts.no_entry {
            let (dpos, dsi) = *global.get(entry_name).with_context(|| {
                format!("entry symbol `{entry_name}` not defined (use --no-entry?)")
            })?;
            let dsym = &included[dpos].symbols[dsi];
            let SymDef::Func { local_index } = dsym.def.as_ref().unwrap() else {
                bail!("entry `{entry_name}` is not a function");
            };
            let t = Target::Func(dpos, *local_index);
            export_list.push((entry_name.to_string(), 0x00, out_func_index(&t)?));
            exported.insert(entry_name.to_string());
        }
        for name in &opts.exports {
            if exported.contains(name) {
                continue;
            }
            let Some(&(dpos, dsi)) = global.get(name) else {
                bail!("--export {name}: symbol not defined");
            };
            let dsym = &included[dpos].symbols[dsi];
            match dsym.def.as_ref().unwrap() {
                SymDef::Func { local_index } => {
                    let t = Target::Func(dpos, *local_index);
                    export_list.push((name.clone(), 0x00, out_func_index(&t)?));
                }
                other => bail!("--export {name}: unsupported symbol kind {other:?}"),
            }
            exported.insert(name.clone());
        }
    }
    drop(resolve);
    drop(out_func_index);
    drop(data_addr);
    // Apply the patches now (mutable pass).
    for pos in 0..included.len() {
        for (off, val) in std::mem::take(&mut code_patches[pos]) {
            let name = included[pos].name.clone();
            apply_patch(&mut included[pos].code_payload, off, val)
                .with_context(|| format!("{name}: code reloc @{off}"))?;
        }
        for (off, val) in std::mem::take(&mut data_patches[pos]) {
            let name = included[pos].name.clone();
            apply_patch(&mut included[pos].data_payload, off, val)
                .with_context(|| format!("{name}: data reloc @{off}"))?;
        }
    }

    mark("apply", &mut t_last);

    // ---- emit ----
    let mut e = Emitter::new();

    // type
    let mut p = Vec::new();
    uleb(&mut p, out_types.len() as u64);
    for t in &out_types {
        p.push(0x60);
        uleb(&mut p, t.params.len() as u64);
        for &v in &t.params {
            p.push(valtype_byte(v));
        }
        uleb(&mut p, t.results.len() as u64);
        for &v in &t.results {
            p.push(valtype_byte(v));
        }
    }
    e.section(1, &p);

    // import (runtime function imports only)
    if !runtime_imports.is_empty() {
        let mut p = Vec::new();
        uleb(&mut p, runtime_imports.len() as u64);
        for (i, (m, f, _)) in runtime_imports.iter().enumerate() {
            write_name(&mut p, m);
            write_name(&mut p, f);
            p.push(0x00);
            uleb(&mut p, rt_import_types[i] as u64);
        }
        e.section(2, &p);
    }

    // function
    let mut p = Vec::new();
    let ctors_present = ctors_index < next;
    let n_defined = included.iter().map(|o| o.func_type_idx.len() as u64).sum::<u64>()
        + stub_types.len() as u64
        + if ctors_present { 1 } else { 0 };
    uleb(&mut p, n_defined);
    for (pos, obj) in included.iter().enumerate() {
        for &ti in &obj.func_type_idx {
            uleb(&mut p, obj_type_map[pos][ti as usize] as u64);
        }
    }
    for &t in &stub_out_types {
        uleb(&mut p, t as u64);
    }
    if ctors_present {
        uleb(&mut p, void_ty as u64);
    }
    e.section(3, &p);

    // table
    let mut p = Vec::new();
    uleb(&mut p, 1);
    p.push(0x70); // funcref
    p.push(0x01); // min+max
    uleb(&mut p, (table_slots.len() + 1) as u64);
    uleb(&mut p, (table_slots.len() + 1) as u64);
    e.section(4, &p);

    // memory
    let mut p = Vec::new();
    uleb(&mut p, 1);
    p.push(0x00); // min only
    uleb(&mut p, mem_min);
    e.section(5, &p);

    // globals: __stack_pointer + synthetics (__memory_base/__table_base/__tls_*)
    let mut p = Vec::new();
    uleb(&mut p, 1 + synth_globals.len() as u64);
    p.push(0x7f);
    p.push(0x01); // i32 mutable
    p.push(0x41); // i32.const
    sleb(&mut p, sp_init as i64);
    p.push(0x0b);
    for &(value, mutable) in &synth_globals {
        p.push(0x7f);
        p.push(if mutable { 0x01 } else { 0x00 });
        p.push(0x41);
        sleb(&mut p, value as i64);
        p.push(0x0b);
    }
    e.section(6, &p);

    // export (resolved earlier, before the mutable patch pass)
    let mut p = Vec::new();
    uleb(&mut p, export_list.len() as u64);
    for (name, kind, idx) in &export_list {
        write_name(&mut p, name);
        p.push(*kind);
        uleb(&mut p, *idx as u64);
    }
    e.section(7, &p);

    // elem: active, table 0, offset 1, func indices
    if !table_slots.is_empty() {
        let mut p = Vec::new();
        uleb(&mut p, 1); // one segment
        p.push(0x00); // active, table 0, funcref via kind 0
        p.push(0x41);
        sleb(&mut p, 1);
        p.push(0x0b);
        uleb(&mut p, table_slots.len() as u64);
        for &fi in &table_slots {
            uleb(&mut p, fi as u64);
        }
        e.section(9, &p);
    }

    // datacount (segments across all objects)
    let n_segs: u64 = included.iter().map(|o| o.segments.len() as u64).sum();
    if n_segs > 0 {
        let mut p = Vec::new();
        uleb(&mut p, n_segs);
        e.section(12, &p);
    }

    // code
    let mut p = Vec::new();
    uleb(&mut p, n_defined);
    for obj in &included {
        for &(start, len) in &obj.code_body_ranges {
            uleb(&mut p, len as u64);
            p.extend_from_slice(&obj.code_payload[start as usize..(start + len) as usize]);
        }
    }
    for _ in &stub_types {
        // body: no locals; unreachable; end
        let body = [0x00u8, 0x00, 0x0b];
        uleb(&mut p, body.len() as u64);
        p.extend_from_slice(&body);
    }
    if ctors_present {
        let mut body = vec![0x00u8]; // no locals
        for &(_prio, fi) in &ctors_calls {
            body.push(0x10);
            uleb(&mut body, fi as u64);
        }
        body.push(0x0b);
        uleb(&mut p, body.len() as u64);
        p.extend_from_slice(&body);
    }
    e.section(10, &p);

    // data
    if n_segs > 0 {
        let mut p = Vec::new();
        uleb(&mut p, n_segs);
        for (pos, obj) in included.iter().enumerate() {
            for (si, seg) in obj.segments.iter().enumerate() {
                p.push(0x00);
                p.push(0x41);
                sleb(&mut p, seg_addrs[pos][si] as i64);
                p.push(0x0b);
                uleb(&mut p, seg.len as u64);
                let s = seg.payload_off as usize;
                p.extend_from_slice(&obj.data_payload[s..s + seg.len as usize]);
            }
        }
        e.section(11, &p);
    }

    mark("emit", &mut t_last);
    if timings_on {
        let mut line = format!("riwl-timing: total_ms={}", t_start.elapsed().as_millis());
        for (name, ms) in &phase_ms {
            line.push_str(&format!(" {name}_ms={ms}"));
        }
        eprintln!("{line}");
    }

    let _ = Linked { objects: included };
    Ok(e.out)
}

fn clone_object(o: &InputObject) -> InputObject {
    InputObject {
        name: o.name.clone(),
        types: o.types.clone(),
        func_imports: o.func_imports.clone(),
        global_imports: o.global_imports.clone(),
        table_imports: o.table_imports.clone(),
        func_type_idx: o.func_type_idx.clone(),
        code_payload: o.code_payload.clone(),
        code_body_ranges: o.code_body_ranges.clone(),
        data_payload: o.data_payload.clone(),
        segments: o.segments.clone(),
        symbols: o.symbols.clone(),
        code_relocs: o.code_relocs.clone(),
        data_relocs: o.data_relocs.clone(),
        init_funcs: o.init_funcs.clone(),
    }
}

#[derive(Copy, Clone)]
enum PatchVal {
    U(u32),
    S(i32),
    I32(u32),
}

fn apply_patch(buf: &mut [u8], off: usize, val: PatchVal) -> Result<()> {
    match val {
        PatchVal::U(v) => {
            ensure!(off + 5 <= buf.len(), "patch overruns section");
            patch_uleb5(buf, off, v);
        }
        PatchVal::S(v) => {
            ensure!(off + 5 <= buf.len(), "patch overruns section");
            patch_sleb5(buf, off, v);
        }
        PatchVal::I32(v) => {
            ensure!(off + 4 <= buf.len(), "patch overruns section");
            patch_le32(buf, off, v);
        }
    }
    Ok(())
}
