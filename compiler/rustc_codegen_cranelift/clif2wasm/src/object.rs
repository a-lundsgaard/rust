//! Relocatable wasm object emission (Phase B1) — the LLVM tool-conventions format:
//! <https://github.com/WebAssembly/tool-conventions/blob/main/Linking.md>.
//!
//! Where `module.rs` assembles one COMPLETE wasm module (absolute data addresses,
//! stubs for anything undefined), this emits a LINKABLE object: undefined symbols
//! stay undefined (function imports from `env`), data lands in named segments placed
//! by the linker, and a `linking` custom section carries the symbol table.
//!
//! Emission strategy: waffle's per-function backend is private, so we build ONE
//! waffle module whose entity layout equals the object's index space —
//! memory 0 = imported `__linear_memory`, global 0 = imported `__stack_pointer`,
//! table 0 = imported `__indirect_function_table`, function imports first — call
//! `to_wasm_bytes()` once, harvest each function's body bytes out of its code
//! section with wasmparser (immediates are raw entity indices, so they are already
//! correct for the object's index space), and hand-assemble the object file around
//! them. We own every byte, so reloc offsets (B1.b+) fall out of the byte stream.

use std::collections::HashMap;

use anyhow::{bail, ensure, Context, Result};
use cranelift_codegen::ir;
use waffle::{FuncDecl, GlobalData, MemoryData, SignatureData, TableData, Type};

use crate::bridge::{self, ModCtx};
use crate::module::{intern_sig, signature_data};
use crate::DataRelocEntry;

/// Symbol binding for the object's symbol table.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ObjBinding {
    Global,
    Weak,
    Local,
}

/// A function definition: translated CLIF, or a synthesized runtime body
/// (i128 helpers via CLIF synthesis, memory.grow/size directly in waffle IR).
pub enum ObjDef<'a> {
    Clif(&'a ir::Function),
    Runtime(crate::RuntimeFn),
}

/// One function slot, in FuncId order. `def: None` = undefined: it becomes a
/// function import from `env` plus an undefined symbol for the linker to resolve.
pub struct ObjFunc<'a> {
    pub name: &'a str,
    pub binding: ObjBinding,
    pub hidden: bool,
    pub def: Option<ObjDef<'a>>,
    /// Authoritative signature (the declaration's; for defined functions this must
    /// match `def.signature`).
    pub sig: ir::Signature,
    /// Reader-dump gv side table (empty when driven from the Module seam).
    pub gv_symbols: HashMap<u32, String>,
}

pub struct ObjDataDef {
    pub bytes: Vec<u8>,
    /// Linear alignment (power of two); emitted as log2 in SEGMENT_INFO.
    pub align: u32,
    pub relocs: Vec<DataRelocEntry>,
}

/// One data slot, in DataId order. `def: None` = undefined data symbol.
pub struct ObjData<'a> {
    pub name: &'a str,
    pub binding: ObjBinding,
    pub hidden: bool,
    pub def: Option<ObjDataDef>,
}

// --- linking-section constants (tool-conventions Linking.md) ---
const WASM_SYMBOL_TABLE: u8 = 8;
const WASM_SEGMENT_INFO: u8 = 5;
const SYMTAB_FUNCTION: u8 = 0;
const SYMTAB_DATA: u8 = 1;
const SYMTAB_GLOBAL: u8 = 2;
const SYMTAB_TABLE: u8 = 5;
const WASM_SYM_BINDING_WEAK: u32 = 0x01;
const WASM_SYM_BINDING_LOCAL: u32 = 0x02;
const WASM_SYM_VISIBILITY_HIDDEN: u32 = 0x04;
const WASM_SYM_UNDEFINED: u32 = 0x10;

// --- relocation types ---
const R_WASM_FUNCTION_INDEX_LEB: u8 = 0;
const R_WASM_TABLE_INDEX_SLEB: u8 = 1;
const R_WASM_TABLE_INDEX_I32: u8 = 2;
const R_WASM_MEMORY_ADDR_SLEB: u8 = 4;
const R_WASM_MEMORY_ADDR_I32: u8 = 5;
const R_WASM_TYPE_INDEX_LEB: u8 = 6;
const R_WASM_GLOBAL_INDEX_LEB: u8 = 7;
const R_WASM_TABLE_NUMBER_LEB: u8 = 20;

/// One relocation record. `index` is a symbol index for every type we emit except
/// TYPE_INDEX_LEB, where it is the type index itself. `addend` is emitted only for
/// the MEMORY_ADDR_* / FUNCTION_OFFSET_* families.
#[derive(Debug)]
struct Reloc {
    ty: u8,
    offset: u32,
    index: u32,
    addend: Option<i64>,
}

fn emit_reloc_section(w: &mut ObjWriter, name: &str, target_section: u32, relocs: &[Reloc]) {
    let mut p = Vec::new();
    uleb(&mut p, target_section as u64);
    uleb(&mut p, relocs.len() as u64);
    for r in relocs {
        p.push(r.ty);
        uleb(&mut p, r.offset as u64);
        uleb(&mut p, r.index as u64);
        if let Some(a) = r.addend {
            sleb(&mut p, a);
        }
    }
    w.custom(name, &p);
}

/// The target_features `+` set, matching what rustc's LLVM backend stamps on
/// wasm32-wasip1 objects (feature-union checked by wasm-ld).
const TARGET_FEATURES: &[&str] = &[
    "bulk-memory",
    "bulk-memory-opt",
    "call-indirect-overlong",
    "multivalue",
    "mutable-globals",
    "nontrapping-fptoint",
    "reference-types",
    "sign-ext",
];

// --- byte-stream helpers ---
fn uleb(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            break;
        }
        out.push(b | 0x80);
    }
}

fn sleb(out: &mut Vec<u8>, mut v: i64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        let sign = b & 0x40 != 0;
        if (v == 0 && !sign) || (v == -1 && sign) {
            out.push(b);
            break;
        }
        out.push(b | 0x80);
    }
}

/// Exactly-5-byte ULEB — reloc sites must be padded so the linker can patch any
/// u32 in place.
fn uleb_padded5(out: &mut Vec<u8>, v: u32) {
    out.push((v & 0x7f) as u8 | 0x80);
    out.push(((v >> 7) & 0x7f) as u8 | 0x80);
    out.push(((v >> 14) & 0x7f) as u8 | 0x80);
    out.push(((v >> 21) & 0x7f) as u8 | 0x80);
    out.push(((v >> 28) & 0x0f) as u8);
}

/// Exactly-5-byte SLEB (i32 range) for SLEB-typed reloc sites.
fn sleb_padded5(out: &mut Vec<u8>, v: i32) {
    out.push((v & 0x7f) as u8 | 0x80);
    out.push(((v >> 7) & 0x7f) as u8 | 0x80);
    out.push(((v >> 14) & 0x7f) as u8 | 0x80);
    out.push(((v >> 21) & 0x7f) as u8 | 0x80);
    out.push(((v >> 28) & 0x7f) as u8); // arithmetic shift keeps the sign bits
}

fn write_name(out: &mut Vec<u8>, s: &str) {
    uleb(out, s.len() as u64);
    out.extend_from_slice(s.as_bytes());
}

fn valtype_byte(t: Type) -> Result<u8> {
    Ok(match t {
        Type::I32 => 0x7f,
        Type::I64 => 0x7e,
        Type::F32 => 0x7d,
        Type::F64 => 0x7c,
        other => bail!("unsupported value type {other} in object emission"),
    })
}

/// Tracks emitted sections so reloc sections can name their target by ordinal
/// (0-based, counting every section entry in order of appearance).
struct ObjWriter {
    out: Vec<u8>,
    next_section: u32,
}

impl ObjWriter {
    fn new() -> Self {
        ObjWriter { out: b"\0asm\x01\0\0\0".to_vec(), next_section: 0 }
    }

    /// Emit one section; returns its ordinal.
    fn section(&mut self, id: u8, payload: &[u8]) -> u32 {
        self.out.push(id);
        uleb(&mut self.out, payload.len() as u64);
        self.out.extend_from_slice(payload);
        let ord = self.next_section;
        self.next_section += 1;
        ord
    }

    fn custom(&mut self, name: &str, payload: &[u8]) -> u32 {
        let mut p = Vec::with_capacity(name.len() + 1 + payload.len());
        write_name(&mut p, name);
        p.extend_from_slice(payload);
        self.section(0, &p)
    }
}

/// A symbol-table entry, in final symbol-index order (functions, then data, then
/// the synthetic global/table symbols).
enum Sym {
    Func { flags: u32, index: u32, name: Option<String> },
    Data { flags: u32, name: String, def: Option<(u32, u32, u32)> }, // (segment, offset, size)
    Global { flags: u32, index: u32 },
    Table { flags: u32, index: u32 },
}

fn binding_flags(binding: ObjBinding, hidden: bool) -> u32 {
    let mut f = match binding {
        ObjBinding::Global => 0,
        ObjBinding::Weak => WASM_SYM_BINDING_WEAK,
        ObjBinding::Local => WASM_SYM_BINDING_LOCAL,
    };
    if hidden {
        f |= WASM_SYM_VISIBILITY_HIDDEN;
    }
    f
}

/// Translate CLIF functions + data into a relocatable wasm object.
pub fn translate_object(funcs: &[ObjFunc], data: &[ObjData]) -> Result<Vec<u8>> {
    // ---- 1. waffle module scaffold: entity layout == object index space ----
    let mut module = waffle::Module::empty();

    let mem0 = module.memories.push(MemoryData {
        initial_pages: 1,
        maximum_pages: None,
        segments: vec![],
    });
    module.imports.push(waffle::Import {
        module: "env".to_string(),
        name: "__linear_memory".to_string(),
        kind: waffle::ImportKind::Memory(mem0),
    });
    let sp_global = module.globals.push(GlobalData { ty: Type::I32, value: None, mutable: true });
    module.imports.push(waffle::Import {
        module: "env".to_string(),
        name: "__stack_pointer".to_string(),
        kind: waffle::ImportKind::Global(sp_global),
    });
    let table = module.tables.push(TableData {
        ty: Type::FuncRef,
        initial: 0,
        max: None,
        func_elements: None,
    });
    module.imports.push(waffle::Import {
        module: "env".to_string(),
        name: "__indirect_function_table".to_string(),
        kind: waffle::ImportKind::Table(table),
    });

    // Signatures (interned with dedup — type indices are shared by bodies'
    // call_indirect immediates and our emitted type section).
    let mut sig_cache: HashMap<SignatureData, waffle::Signature> = HashMap::new();
    let mut sigs = Vec::with_capacity(funcs.len());
    for f in funcs {
        let sd = signature_data(&f.sig)
            .with_context(|| format!("signature of function `{}`", f.name))?;
        sigs.push(intern_sig(&mut module, &mut sig_cache, sd));
    }

    // Libcall callees (`ExternalName::LibCall`) become undefined imports under
    // their real C names, resolved from compiler_builtins/libc at link time.
    // (Whole-module mode synthesizes helpers instead; and there memmove merges
    // into memcpy — here the names MUST stay distinct: libc memcpy is not
    // overlap-safe.)
    let mut libcall_sigs: HashMap<crate::RuntimeFn, waffle::Signature> = HashMap::new();
    for f in funcs {
        let Some(ObjDef::Clif(clif)) = &f.def else { continue };
        for ext in clif.dfg.ext_funcs.values() {
            let ir::ExternalName::LibCall(lc) = ext.name else { continue };
            let kind = match lc {
                ir::LibCall::Memcpy => crate::RuntimeFn::Memcpy,
                ir::LibCall::Memmove => crate::RuntimeFn::Memmove,
                ir::LibCall::Memset => crate::RuntimeFn::Memset,
                ir::LibCall::Memcmp => crate::RuntimeFn::Memcmp,
                other => bail!("unimplemented libcall callee {other} in `{}`", f.name),
            };
            if libcall_sigs.contains_key(&kind) {
                continue;
            }
            let sd = signature_data(&clif.dfg.signatures[ext.signature])
                .with_context(|| format!("libcall {lc} signature in `{}`", f.name))?;
            libcall_sigs.insert(kind, intern_sig(&mut module, &mut sig_cache, sd));
        }
    }

    // Function entities: imports (undefined + libcalls) FIRST so entity index ==
    // wasm function index, then placeholders for defined functions.
    use waffle::entity::EntityRef as _;
    let mut wfuncs = vec![waffle::Func::invalid(); funcs.len()];
    let mut import_fn_ords: Vec<usize> = Vec::new(); // input ordinal, in import order
    let mut defined_fn_ords: Vec<usize> = Vec::new(); // input ordinal, in defined order
    for (k, f) in funcs.iter().enumerate() {
        if f.def.is_none() {
            let wf = module.funcs.push(FuncDecl::Import(sigs[k], f.name.to_string()));
            module.imports.push(waffle::Import {
                module: "env".to_string(),
                name: f.name.to_string(),
                kind: waffle::ImportKind::Func(wf),
            });
            wfuncs[k] = wf;
            import_fn_ords.push(k);
        }
    }
    // Fixed order for determinism.
    let mut libcalls: Vec<(crate::RuntimeFn, &'static str, waffle::Func, waffle::Signature)> =
        Vec::new();
    let mut libcall_map: HashMap<crate::RuntimeFn, waffle::Func> = HashMap::new();
    for (kind, cname) in [
        (crate::RuntimeFn::Memcpy, "memcpy"),
        (crate::RuntimeFn::Memmove, "memmove"),
        (crate::RuntimeFn::Memset, "memset"),
        (crate::RuntimeFn::Memcmp, "memcmp"),
    ] {
        let Some(&sig) = libcall_sigs.get(&kind) else { continue };
        let wf = module.funcs.push(FuncDecl::Import(sig, cname.to_string()));
        module.imports.push(waffle::Import {
            module: "env".to_string(),
            name: cname.to_string(),
            kind: waffle::ImportKind::Func(wf),
        });
        libcall_map.insert(kind, wf);
        libcalls.push((kind, cname, wf, sig));
    }
    for (k, f) in funcs.iter().enumerate() {
        if f.def.is_some() {
            wfuncs[k] = module.funcs.push(FuncDecl::None);
            defined_fn_ords.push(k);
        }
    }
    // input ordinal -> wasm function index (imports first).
    let fn_wasm_index = |k: usize| wfuncs[k].index() as u32;

    // Pre-scan defined bodies for symbol references and allocate MARKER globals:
    // one per unique (data ordinal, offset) for `symbol_value`, one per
    // address-taken function for `func_addr`. Bodies emit `global.get <marker>`;
    // the splicer rewrites each site into a padded i32.const + reloc. Marker
    // globals are entities only — they are never emitted into any section.
    let mut markers = bridge::ObjectMarkers::default();
    for f in funcs {
        let Some(ObjDef::Clif(clif)) = &f.def else { continue };
        for (_gv, gvd) in clif.global_values.iter() {
            let ir::GlobalValueData::Symbol { name, offset, .. } = gvd else { continue };
            let ir::ExternalName::User(r) = name else { continue };
            let Some(user) = clif.params.user_named_funcs().get(*r) else { continue };
            if user.namespace != 1 {
                continue;
            }
            let ord = user.index;
            ensure!(
                (ord as usize) < data.len(),
                "function `{}` references data ordinal {ord} out of range",
                f.name
            );
            let off: i64 = (*offset).into();
            markers.data.entry((ord, off)).or_insert_with(|| {
                module.globals.push(GlobalData { ty: Type::I32, value: Some(0), mutable: false })
            });
        }
        for block in clif.layout.blocks() {
            for inst in clif.layout.block_insts(block) {
                let ir::InstructionData::FuncAddr { func_ref, .. } = clif.dfg.insts[inst] else {
                    continue;
                };
                let ext = &clif.dfg.ext_funcs[func_ref];
                let ir::ExternalName::User(r) = ext.name else { continue };
                let Some(user) = clif.params.user_named_funcs().get(r) else { continue };
                if user.namespace != 0 {
                    continue;
                }
                let ord = user.index;
                ensure!(
                    (ord as usize) < funcs.len(),
                    "function `{}` takes the address of ordinal {ord} out of range",
                    f.name
                );
                markers.funcs.entry(ord).or_insert_with(|| {
                    module.globals.push(GlobalData { ty: Type::I32, value: Some(0), mutable: false })
                });
            }
        }
    }

    // ---- 2. translate bodies ----
    let empty_addrs: Vec<u32> = Vec::new();
    let empty_by_name: HashMap<String, u32> = HashMap::new();
    let mctx = ModCtx {
        callees: &wfuncs,
        mem0,
        sp_global,
        table,
        data_addrs: &empty_addrs,
        data_by_name: &empty_by_name,
        libcalls: &libcall_map,
        object_markers: Some(&markers),
    };
    for &k in &defined_fn_ords {
        let f = &funcs[k];
        let body = match f.def.as_ref().unwrap() {
            ObjDef::Clif(clif) => {
                let mut sig_map: HashMap<ir::SigRef, waffle::Signature> = HashMap::new();
                for (sig_ref, sig) in clif.dfg.signatures.iter() {
                    let sd = signature_data(sig)
                        .with_context(|| format!("SigRef {sig_ref} in function `{}`", f.name))?;
                    sig_map.insert(sig_ref, intern_sig(&mut module, &mut sig_cache, sd));
                }
                bridge::translate_function(clif, &module, sigs[k], &mctx, &sig_map, &f.gv_symbols)
                    .with_context(|| format!("translating function `{}`", f.name))?
            }
            &ObjDef::Runtime(kind) => {
                if let Some(clif_fn) = crate::runtime_i128::build(kind) {
                    let empty_sigs = HashMap::new();
                    let empty_gvs = HashMap::new();
                    bridge::translate_function(
                        &clif_fn, &module, sigs[k], &mctx, &empty_sigs, &empty_gvs,
                    )
                    .with_context(|| format!("translating runtime `{}`", f.name))?
                } else {
                    crate::module::synth_runtime_body(kind, &module, sigs[k], mem0)
                        .with_context(|| format!("synthesizing runtime `{}`", f.name))?
                }
            }
        };
        body.validate()
            .map_err(|e| anyhow::anyhow!("waffle validate() rejected `{}`:\n{e}", f.name))?;
        module.funcs[wfuncs[k]] = FuncDecl::Body(sigs[k], f.name.to_string(), body);
    }

    // ---- 3. compile via waffle, harvest per-function body bytes ----
    let compiled = module.to_wasm_bytes().context("waffle backend compile")?;
    let bodies = harvest_code_bodies(&compiled, defined_fn_ords.len())?;

    // ---- 4. assemble the object ----
    let mut w = ObjWriter::new();

    // type section: waffle signatures in entity order.
    let mut p = Vec::new();
    uleb(&mut p, module.signatures.len() as u64);
    for (_sig, sd) in module.signatures.entries() {
        p.push(0x60);
        uleb(&mut p, sd.params.len() as u64);
        for &t in &sd.params {
            p.push(valtype_byte(t)?);
        }
        uleb(&mut p, sd.returns.len() as u64);
        for &t in &sd.returns {
            p.push(valtype_byte(t)?);
        }
    }
    w.section(1, &p);

    // import section: memory, stack-pointer global, funcref table, undefined fns,
    // libcalls — function order MUST match entity allocation above.
    let mut p = Vec::new();
    uleb(&mut p, (3 + import_fn_ords.len() + libcalls.len()) as u64);
    write_name(&mut p, "env");
    write_name(&mut p, "__linear_memory");
    p.extend_from_slice(&[0x02, 0x00, 0x01]); // memory, limits: min 1, no max
    write_name(&mut p, "env");
    write_name(&mut p, "__stack_pointer");
    p.extend_from_slice(&[0x03, 0x7f, 0x01]); // global, i32, mutable
    write_name(&mut p, "env");
    write_name(&mut p, "__indirect_function_table");
    p.extend_from_slice(&[0x01, 0x70, 0x00, 0x00]); // table, funcref, limits: min 0
    for &k in &import_fn_ords {
        write_name(&mut p, "env");
        write_name(&mut p, funcs[k].name);
        p.push(0x00); // func
        uleb(&mut p, sigs[k].index() as u64);
    }
    for (_kind, cname, _wf, sig) in &libcalls {
        write_name(&mut p, "env");
        write_name(&mut p, cname);
        p.push(0x00); // func
        uleb(&mut p, sig.index() as u64);
    }
    w.section(2, &p);

    // function section: defined functions' signature indices, in entity order.
    let mut p = Vec::new();
    uleb(&mut p, defined_fn_ords.len() as u64);
    for &k in &defined_fn_ords {
        uleb(&mut p, sigs[k].index() as u64);
    }
    w.section(3, &p);

    // data segments: every DEFINED data object, in input order.
    let segments: Vec<(usize, &ObjDataDef)> = data
        .iter()
        .enumerate()
        .filter_map(|(i, d)| d.def.as_ref().map(|def| (i, def)))
        .collect();

    if !segments.is_empty() {
        let mut p = Vec::new();
        uleb(&mut p, segments.len() as u64);
        w.section(12, &p); // DataCount
    }

    // Symbol index layout (must match the symbol table below): function symbols in
    // input order, then libcall symbols, then data symbols in input order, then
    // __stack_pointer, __indirect_function_table.
    let data_sym_base = (funcs.len() + libcalls.len()) as u32;
    let sp_sym = data_sym_base + data.len() as u32;
    let table_sym = sp_sym + 1;
    let mut fn_sym_of_wasm_index = vec![u32::MAX; funcs.len() + libcalls.len()];
    for k in 0..funcs.len() {
        fn_sym_of_wasm_index[fn_wasm_index(k) as usize] = k as u32;
    }
    for (j, (_kind, _cname, wf, _sig)) in libcalls.iter().enumerate() {
        fn_sym_of_wasm_index[wf.index()] = (funcs.len() + j) as u32;
    }
    // marker global entity index -> (reloc type, symbol index, addend).
    let mut marker_relocs: HashMap<u32, (u8, u32, Option<i64>)> = HashMap::new();
    for (&(ord, off), g) in &markers.data {
        marker_relocs.insert(
            g.index() as u32,
            (R_WASM_MEMORY_ADDR_SLEB, data_sym_base + ord, Some(off)),
        );
    }
    for (&ord, g) in &markers.funcs {
        marker_relocs.insert(g.index() as u32, (R_WASM_TABLE_INDEX_SLEB, ord, None));
    }

    // code section: splice each body (padded reloc-site immediates + per-body
    // relocs), then lay out the section, rebasing reloc offsets onto the payload.
    let mut code_relocs: Vec<Reloc> = Vec::new();
    let mut p = Vec::new();
    uleb(&mut p, bodies.len() as u64);
    for (i, body) in bodies.iter().enumerate() {
        let (spliced, relocs) =
            splice_body(body, &fn_sym_of_wasm_index, sp_sym, table_sym, &marker_relocs)
                .with_context(|| {
                    format!("splicing function `{}`", funcs[defined_fn_ords[i]].name)
                })?;
        uleb(&mut p, spliced.len() as u64);
        let body_base = p.len() as u32;
        for mut r in relocs {
            r.offset += body_base;
            code_relocs.push(r);
        }
        p.extend_from_slice(&spliced);
    }
    let code_ord = w.section(10, &p);

    // data section: active segments with placeholder offsets (placement is the
    // linker's job, guided by SEGMENT_INFO); pointer slots become reloc.DATA
    // entries against the target's symbol, slots left as-is for the linker.
    let mut data_relocs: Vec<Reloc> = Vec::new();
    let mut data_ord = None;
    if !segments.is_empty() {
        let mut p = Vec::new();
        uleb(&mut p, segments.len() as u64);
        for (di, def) in &segments {
            p.push(0x00); // active, memory 0
            p.extend_from_slice(&[0x41, 0x00, 0x0b]); // i32.const 0; end
            uleb(&mut p, def.bytes.len() as u64);
            let seg_base = p.len() as u32;
            for r in &def.relocs {
                ensure!(
                    (r.offset as usize) + 4 <= def.bytes.len(),
                    "data `{}` reloc at {} overruns object of {} bytes",
                    data[*di].name,
                    r.offset,
                    def.bytes.len()
                );
                match r.target {
                    crate::DataRelocTarget::Data(ord) => {
                        ensure!((ord as usize) < data.len(), "data reloc ordinal {ord} out of range");
                        data_relocs.push(Reloc {
                            ty: R_WASM_MEMORY_ADDR_I32,
                            offset: seg_base + r.offset,
                            index: data_sym_base + ord,
                            addend: Some(r.addend),
                        });
                    }
                    crate::DataRelocTarget::Func(ord) => {
                        ensure!((ord as usize) < funcs.len(), "func reloc ordinal {ord} out of range");
                        ensure!(r.addend == 0, "function-address reloc with addend {}", r.addend);
                        data_relocs.push(Reloc {
                            ty: R_WASM_TABLE_INDEX_I32,
                            offset: seg_base + r.offset,
                            index: ord,
                            addend: None,
                        });
                    }
                }
            }
            p.extend_from_slice(&def.bytes);
        }
        data_ord = Some(w.section(11, &p));
    }

    // ---- symbol table ----
    let mut syms: Vec<Sym> = Vec::new();
    for (k, f) in funcs.iter().enumerate() {
        let mut flags = binding_flags(f.binding, f.hidden);
        let name = if f.def.is_some() {
            Some(f.name.to_string())
        } else {
            flags |= WASM_SYM_UNDEFINED;
            None // name comes from the import entry
        };
        syms.push(Sym::Func { flags, index: fn_wasm_index(k), name });
    }
    for (_kind, _cname, wf, _sig) in &libcalls {
        // Undefined; name comes from the import entry.
        syms.push(Sym::Func { flags: WASM_SYM_UNDEFINED, index: wf.index() as u32, name: None });
    }
    // data-input ordinal -> segment index, for defined data.
    let mut seg_index: HashMap<usize, u32> = HashMap::new();
    for (si, (i, _)) in segments.iter().enumerate() {
        seg_index.insert(*i, si as u32);
    }
    for (i, d) in data.iter().enumerate() {
        let mut flags = binding_flags(d.binding, d.hidden);
        let def = match &d.def {
            Some(def) => Some((seg_index[&i], 0u32, def.bytes.len() as u32)),
            None => {
                flags |= WASM_SYM_UNDEFINED;
                None
            }
        };
        syms.push(Sym::Data { flags, name: d.name.to_string(), def });
    }
    // Synthetic linker-provided references (always imported by this writer).
    syms.push(Sym::Global { flags: WASM_SYM_UNDEFINED, index: 0 });
    syms.push(Sym::Table { flags: WASM_SYM_UNDEFINED, index: 0 });

    // linking section: version 2, SYMBOL_TABLE + SEGMENT_INFO subsections.
    let mut link = Vec::new();
    uleb(&mut link, 2); // version

    let mut st = Vec::new();
    uleb(&mut st, syms.len() as u64);
    for s in &syms {
        match s {
            Sym::Func { flags, index, name } => {
                st.push(SYMTAB_FUNCTION);
                uleb(&mut st, *flags as u64);
                uleb(&mut st, *index as u64);
                if let Some(n) = name {
                    write_name(&mut st, n);
                }
            }
            Sym::Data { flags, name, def } => {
                st.push(SYMTAB_DATA);
                uleb(&mut st, *flags as u64);
                write_name(&mut st, name);
                if let Some((seg, off, size)) = def {
                    uleb(&mut st, *seg as u64);
                    uleb(&mut st, *off as u64);
                    uleb(&mut st, *size as u64);
                }
            }
            Sym::Global { flags, index } => {
                st.push(SYMTAB_GLOBAL);
                uleb(&mut st, *flags as u64);
                uleb(&mut st, *index as u64);
            }
            Sym::Table { flags, index } => {
                st.push(SYMTAB_TABLE);
                uleb(&mut st, *flags as u64);
                uleb(&mut st, *index as u64);
            }
        }
    }
    link.push(WASM_SYMBOL_TABLE);
    uleb(&mut link, st.len() as u64);
    link.extend_from_slice(&st);

    if !segments.is_empty() {
        let mut si = Vec::new();
        uleb(&mut si, segments.len() as u64);
        for (i, def) in &segments {
            ensure!(
                def.align.is_power_of_two(),
                "data `{}` align {} not a power of two",
                data[*i].name,
                def.align
            );
            write_name(&mut si, data[*i].name);
            uleb(&mut si, def.align.trailing_zeros() as u64); // log2
            uleb(&mut si, 0); // flags
        }
        link.push(WASM_SEGMENT_INFO);
        uleb(&mut link, si.len() as u64);
        link.extend_from_slice(&si);
    }
    w.custom("linking", &link);

    if !code_relocs.is_empty() {
        emit_reloc_section(&mut w, "reloc.CODE", code_ord, &code_relocs);
    }
    if !data_relocs.is_empty() {
        emit_reloc_section(&mut w, "reloc.DATA", data_ord.expect("data section exists"), &data_relocs);
    }

    // target_features.
    let mut tf = Vec::new();
    uleb(&mut tf, TARGET_FEATURES.len() as u64);
    for f in TARGET_FEATURES {
        tf.push(b'+');
        write_name(&mut tf, f);
    }
    w.custom("target_features", &tf);

    Ok(w.out)
}

/// Splice one harvested body: copy bytes verbatim, except at reloc-bearing
/// instructions, which are re-emitted with 5-byte padded LEB immediates and
/// recorded as relocations (offsets relative to the START OF THIS BODY's bytes).
///
/// Reloc-bearing sites: `call` (FUNCTION_INDEX_LEB via the function's symbol),
/// `call_indirect` (TYPE_INDEX_LEB on the type + TABLE_NUMBER_LEB on the table),
/// `global.get/set` of the stack pointer (GLOBAL_INDEX_LEB). Any other global
/// index or a `ref.func` is a bug at this stage — B1.c introduces marker globals.
fn splice_body(
    body: &[u8],
    fn_sym_of_wasm_index: &[u32],
    sp_sym: u32,
    table_sym: u32,
    marker_relocs: &HashMap<u32, (u8, u32, Option<i64>)>,
) -> Result<(Vec<u8>, Vec<Reloc>)> {
    use wasmparser::Operator as Op;
    let mut out = Vec::with_capacity(body.len() + 32);
    let mut relocs = Vec::new();
    let fb = wasmparser::FunctionBody::new(wasmparser::BinaryReader::new(body, 0));
    let mut reader = fb.get_operators_reader()?;
    let mut copied_to = 0usize; // source offset up to which bytes are flushed
    while !reader.eof() {
        let op_start = reader.original_position();
        let op = reader.read()?;
        let op_end = reader.original_position();
        match op {
            Op::Call { function_index } => {
                out.extend_from_slice(&body[copied_to..op_start]);
                out.push(0x10);
                let sym = *fn_sym_of_wasm_index
                    .get(function_index as usize)
                    .ok_or_else(|| anyhow::anyhow!("call to unknown function index {function_index}"))?;
                relocs.push(Reloc {
                    ty: R_WASM_FUNCTION_INDEX_LEB,
                    offset: out.len() as u32,
                    index: sym,
                    addend: None,
                });
                uleb_padded5(&mut out, function_index);
                copied_to = op_end;
            }
            Op::CallIndirect { type_index, table_index } => {
                out.extend_from_slice(&body[copied_to..op_start]);
                out.push(0x11);
                relocs.push(Reloc {
                    ty: R_WASM_TYPE_INDEX_LEB,
                    offset: out.len() as u32,
                    index: type_index, // type index, not a symbol
                    addend: None,
                });
                uleb_padded5(&mut out, type_index);
                ensure!(table_index == 0, "call_indirect through table {table_index}");
                relocs.push(Reloc {
                    ty: R_WASM_TABLE_NUMBER_LEB,
                    offset: out.len() as u32,
                    index: table_sym,
                    addend: None,
                });
                uleb_padded5(&mut out, table_index);
                copied_to = op_end;
            }
            Op::GlobalGet { global_index } => {
                out.extend_from_slice(&body[copied_to..op_start]);
                if global_index == 0 {
                    out.push(0x23);
                    relocs.push(Reloc {
                        ty: R_WASM_GLOBAL_INDEX_LEB,
                        offset: out.len() as u32,
                        index: sp_sym,
                        addend: None,
                    });
                    uleb_padded5(&mut out, 0);
                } else if let Some(&(ty, sym, addend)) = marker_relocs.get(&global_index) {
                    // Marker site: becomes a padded i32.const carrying a
                    // MEMORY_ADDR_SLEB / TABLE_INDEX_SLEB relocation.
                    out.push(0x41);
                    relocs.push(Reloc { ty, offset: out.len() as u32, index: sym, addend });
                    sleb_padded5(&mut out, 0);
                } else {
                    bail!("global.get {global_index} is neither the stack pointer nor a marker");
                }
                copied_to = op_end;
            }
            Op::GlobalSet { global_index } => {
                ensure!(global_index == 0, "global.set {global_index} in object mode");
                out.extend_from_slice(&body[copied_to..op_start]);
                out.push(0x24);
                relocs.push(Reloc {
                    ty: R_WASM_GLOBAL_INDEX_LEB,
                    offset: out.len() as u32,
                    index: sp_sym,
                    addend: None,
                });
                uleb_padded5(&mut out, 0);
                copied_to = op_end;
            }
            Op::RefFunc { .. } => bail!("ref.func in object mode is unsupported (unexpected from our lowering)"),
            _ => {}
        }
    }
    out.extend_from_slice(&body[copied_to..]);
    Ok((out, relocs))
}

/// Pull each function's body bytes (locals + instructions, no size prefix) out of a
/// complete wasm module's code section.
fn harvest_code_bodies(wasm: &[u8], expect: usize) -> Result<Vec<Vec<u8>>> {
    use wasmparser::{Parser, Payload};
    let mut bodies = Vec::with_capacity(expect);
    for payload in Parser::new(0).parse_all(wasm) {
        if let Payload::CodeSectionEntry(body) = payload.context("parsing waffle output")? {
            let range = body.range();
            bodies.push(wasm[range.start..range.end].to_vec());
        }
    }
    ensure!(
        bodies.len() == expect,
        "harvested {} bodies from waffle output, expected {expect}",
        bodies.len()
    );
    Ok(bodies)
}
