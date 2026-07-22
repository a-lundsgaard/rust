//! Module assembly: signatures, function declarations, memory, stack-pointer global,
//! funcref table, export, emission.

use std::collections::HashMap;

use anyhow::{bail, ensure, Context, Result};
use cranelift_codegen::ir;
use waffle::{
    Export, ExportKind, FuncDecl, GlobalData, MemoryData, MemorySegment, SignatureData, TableData,
    Type,
};

use crate::bridge::{self, ModCtx};
use crate::{expand_type, NamedClifData, NamedClifFunc, RuntimeFn};
use waffle::{FunctionBody, Operator, Terminator};

/// One 64KiB wasm page; the shadow stack starts at its top and grows down.
const STACK_TOP: u64 = 0x1_0000;
/// Statics are bump-allocated upward from here (0x0–0xFFF stays as a null guard);
/// they live below the shadow stack, which grows down from STACK_TOP.
const DATA_BASE: u32 = 0x1000;

pub fn translate_module(
    funcs: &[NamedClifFunc],
    data: &[NamedClifData],
    exports: &[&str],
) -> Result<Vec<u8>> {
    translate_module_impl(funcs, data, exports, &Default::default(), &Default::default())
}

/// `runtime_overrides`: slice positions whose BODY is synthesized (the entry's CLIF
/// function contributes only its signature) — the named-libcall path (e.g. an
/// undefined `memcmp` import replaced by the real byte-compare loop).
pub fn translate_module_impl(
    funcs: &[NamedClifFunc],
    data: &[NamedClifData],
    exports: &[&str],
    runtime_overrides: &std::collections::HashMap<usize, RuntimeFn>,
    imports: &std::collections::HashMap<usize, String>,
) -> Result<Vec<u8>> {
    let mut module = waffle::Module::empty();

    // Lay out static data objects: bump-allocate from DATA_BASE honoring per-object
    // alignment; one memory segment each.
    let mut data_addrs = Vec::with_capacity(data.len());
    let mut data_by_name: std::collections::HashMap<String, u32> = Default::default();
    let mut segments = Vec::with_capacity(data.len());
    let mut cur = DATA_BASE;
    for d in data {
        ensure!(d.align.is_power_of_two(), "data `{}` align {} not a power of two", d.name, d.align);
        cur = (cur + d.align - 1) & !(d.align - 1);
        data_addrs.push(cur);
        data_by_name.insert(d.name.to_string(), cur);
        segments.push(MemorySegment { offset: cur as usize, data: d.bytes.clone() });
        cur += d.bytes.len() as u32;
    }
    // Shadow stack lives ABOVE the statics: 64 KiB, 16-aligned base; memory sized to
    // fit (one page minimum — small modules keep the historical 0x10000 stack top).
    let statics_end = ((cur + 15) & !15) as u64;
    let stack_top = core::cmp::max(STACK_TOP, statics_end + 0x1_0000);
    let pages = (stack_top + 0xFFFF) / 0x1_0000;

    // Apply data relocations now that every address is known: pointer slots get the
    // target's linear-memory address (data) or funcref-table index (function).
    for (k, d) in data.iter().enumerate() {
        for r in &d.relocs {
            let value: i64 = match r.target {
                crate::DataRelocTarget::Data(ord) => {
                    *data_addrs.get(ord as usize).with_context(|| {
                        format!("data `{}` reloc: data ordinal {ord} out of range", d.name)
                    })? as i64
                }
                // Table slot 0 is null; function k lives at k+1 (matches func_addr).
                crate::DataRelocTarget::Func(ord) => {
                    ensure!(
                        (ord as usize) < funcs.len(),
                        "data `{}` reloc: function ordinal {ord} out of range",
                        d.name
                    );
                    ord as i64 + 1
                }
            };
            let value = value + r.addend;
            let off = r.offset as usize;
            let bytes = &mut segments[k].data;
            ensure!(
                off + 4 <= bytes.len(),
                "data `{}` reloc at {off} overruns object of {} bytes",
                d.name,
                bytes.len()
            );
            bytes[off..off + 4].copy_from_slice(&(value as u32).to_le_bytes());
        }
    }

    // One linear memory so loads/stores validate; statics are its initial image.
    let mem0 = module.memories.push(MemoryData {
        initial_pages: pages as usize,
        maximum_pages: None,
        segments,
    });

    // The shadow-stack pointer (plan §8): a mutable i32 global, decremented per frame.
    let sp_global = module.globals.push(GlobalData {
        ty: Type::I32,
        value: Some(stack_top),
        mutable: true,
    });

    // Intern signatures with dedup: entry signatures now, per-function SigRefs below.
    let mut sig_cache: HashMap<SignatureData, waffle::Signature> = HashMap::new();

    // Declare every signature and a FuncDecl placeholder up front, so waffle `Func`
    // indices exist before any body is translated (direct calls need them).
    let lenient_env = std::env::var_os("CLIF2WASM_LENIENT").is_some_and(|v| v == "1");
    let mut sigs = Vec::with_capacity(funcs.len());
    let mut wfuncs = Vec::with_capacity(funcs.len());
    // Slots whose SIGNATURE is untranslatable (f16/f128): lenient mode gives them a
    // dummy () -> () sig and an unconditional trap stub — callers reaching for them
    // are themselves f16/f128 paths that lenient-stub on their own call lowering.
    let mut force_stub: std::collections::HashSet<usize> = Default::default();
    for (k, nf) in funcs.iter().enumerate() {
        let data = match signature_data(&nf.func.signature) {
            Ok(d) => d,
            Err(e) if lenient_env => {
                eprintln!(
                    "clif2wasm-gap: signature stubbed: {} ({})",
                    nf.name,
                    first_line(&format!("{e:#}"))
                );
                force_stub.insert(k);
                SignatureData { params: vec![], returns: vec![] }
            }
            Err(e) => {
                return Err(e).with_context(|| format!("signature of function `{}`", nf.name))
            }
        };
        sigs.push(intern_sig(&mut module, &mut sig_cache, data));
    }
    // wasm requires imported functions FIRST in the index space: allocate import
    // FuncDecls before any defined placeholder, keeping `wfuncs` slice-indexed.
    use waffle::entity::EntityRef as _;
    wfuncs.resize(funcs.len(), waffle::Func::invalid());
    for (k, import_name) in imports {
        let f = module.funcs.push(FuncDecl::Import(sigs[*k], import_name.clone()));
        module.imports.push(waffle::Import {
            module: "env".to_string(),
            name: import_name.clone(),
            kind: waffle::ImportKind::Func(f),
        });
        wfuncs[*k] = f;
    }
    for k in 0..funcs.len() {
        if !imports.contains_key(&k) {
            wfuncs[k] = module.funcs.push(FuncDecl::None);
        }
    }

    // One funcref table for call_indirect: slot 0 is null, function k lives at k+1
    // (so a zero function pointer never silently dispatches).
    let table = {
        use waffle::entity::EntityRef;
        let mut elements = vec![waffle::Func::invalid()];
        elements.extend(wfuncs.iter().copied());
        module.tables.push(TableData {
            ty: Type::FuncRef,
            initial: elements.len() as u64,
            max: Some(elements.len() as u64),
            func_elements: Some(elements),
        })
    };

    // Pre-scan for `ExternalName::LibCall` callees inside any function and synthesize
    // one helper per needed runtime kind (appended after the user functions; they take
    // no funcref-table slots — nothing takes their address).
    let mut libcalls: std::collections::HashMap<RuntimeFn, waffle::Func> = Default::default();
    let lenient = std::env::var_os("CLIF2WASM_LENIENT").is_some_and(|v| v == "1");
    for nf in funcs {
        for ext in nf.func.dfg.ext_funcs.values() {
            if let ir::ExternalName::LibCall(lc) = ext.name {
                let kind = match libcall_kind(lc) {
                    Ok(k) => k,
                    Err(e) if lenient => {
                        // The referencing body will fail resolve_callee and get
                        // lenient-stubbed; just report here.
                        eprintln!("clif2wasm-gap: libcall {lc} in `{}` ({e:#})", nf.name);
                        continue;
                    }
                    Err(e) => return Err(e),
                };
                if libcalls.contains_key(&kind) {
                    continue;
                }
                let data = signature_data(&nf.func.dfg.signatures[ext.signature])
                    .with_context(|| format!("libcall {lc} signature"))?;
                let sig = intern_sig(&mut module, &mut sig_cache, data);
                let body = synth_runtime_body(kind, &module, sig, mem0)?;
                let f = module
                    .funcs
                    .push(FuncDecl::Body(sig, format!("__clif2wasm_{kind:?}"), body));
                libcalls.insert(kind, f);
            }
        }
    }

    let mctx = ModCtx {
        callees: &wfuncs,
        mem0,
        sp_global,
        table,
        data_addrs: &data_addrs,
        data_by_name: &data_by_name,
        libcalls: &libcalls,
        object_markers: None,
    };

    for (k, nf) in funcs.iter().enumerate() {
        // Import slots have no body.
        if imports.contains_key(&k) {
            continue;
        }
        // Runtime-override slots: synthesize the body from the declared signature —
        // either directly in waffle IR (memory helpers) or as CLIF fed through our
        // own translator (the i128 division/mulo family).
        if let Some(&kind) = runtime_overrides.get(&k) {
            let body = if let Some(clif_fn) = crate::runtime_i128::build(kind) {
                let empty_sigs = std::collections::HashMap::new();
                let empty_gvs = std::collections::HashMap::new();
                bridge::translate_function(&clif_fn, &module, sigs[k], &mctx, &empty_sigs, &empty_gvs)
                    .with_context(|| format!("translating runtime `{}`", nf.name))?
            } else {
                synth_runtime_body(kind, &module, sigs[k], mem0)
                    .with_context(|| format!("synthesizing runtime `{}`", nf.name))?
            };
            body.validate().map_err(|e| anyhow::anyhow!("runtime `{}`: {e}", nf.name))?;
            module.funcs[wfuncs[k]] = FuncDecl::Body(sigs[k], nf.name.to_string(), body);
            continue;
        }
        if force_stub.contains(&k) {
            let mut stub = waffle::FunctionBody::new(&module, sigs[k]);
            let entry = stub.entry;
            stub.set_terminator(entry, Terminator::Unreachable);
            module.funcs[wfuncs[k]] = FuncDecl::Body(sigs[k], nf.name.to_string(), stub);
            continue;
        }
        // Intern every signature this function references (call_indirect needs them).
        // Lenient mode skips unconvertible ones — bodies using them stub themselves.
        let mut sig_map: HashMap<ir::SigRef, waffle::Signature> = HashMap::new();
        for (sig_ref, sig) in nf.func.dfg.signatures.iter() {
            let data = match signature_data(sig) {
                Ok(d) => d,
                Err(_) if lenient_env => continue,
                Err(e) => {
                    return Err(e)
                        .with_context(|| format!("SigRef {sig_ref} in function `{}`", nf.name))
                }
            };
            sig_map.insert(sig_ref, intern_sig(&mut module, &mut sig_cache, data));
        }

        let translated = bridge::translate_function(
            nf.func,
            &module,
            sigs[k],
            &mctx,
            &sig_map,
            &nf.gv_symbols,
        )
        .with_context(|| format!("translating function `{}`", nf.name))
        .and_then(|body| {
            body.validate().map_err(|e| {
                anyhow::anyhow!(
                    "waffle validate() rejected `{}`:\n{}\nCLIF was:\n{}",
                    nf.name,
                    e,
                    nf.func.display()
                )
            })?;
            Ok(body)
        });
        let body = match translated {
            Ok(body) => body,
            // Lenient (sysroot) mode: an untranslatable BODY becomes a trap stub with
            // a reported gap line — sysroot rlibs are consumed for metadata; stubbed
            // bodies only matter if actually called (then they trap loudly).
            Err(e) if std::env::var_os("CLIF2WASM_LENIENT").is_some_and(|v| v == "1") => {
                eprintln!(
                    "clif2wasm-gap: body stubbed: {} ({})",
                    nf.name,
                    first_line(&format!("{e:#}"))
                );
                let mut stub = waffle::FunctionBody::new(&module, sigs[k]);
                let entry = stub.entry;
                stub.set_terminator(entry, Terminator::Unreachable);
                stub
            }
            Err(e) => return Err(e),
        };
        module.funcs[wfuncs[k]] = FuncDecl::Body(sigs[k], nf.name.to_string(), body);
    }

    for export in exports {
        let Some(entry) = funcs.iter().position(|f| f.name == *export) else {
            bail!("export `{export}` is not among the provided functions");
        };
        module.exports.push(Export {
            name: export.to_string(),
            kind: ExportKind::Func(wfuncs[entry]),
        });
    }

    module.to_wasm_bytes().context("waffle backend compile")
}

fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or(s)
}

fn libcall_kind(lc: ir::LibCall) -> Result<RuntimeFn> {
    Ok(match lc {
        // memory.copy is overlap-safe, so one helper serves both.
        ir::LibCall::Memcpy | ir::LibCall::Memmove => RuntimeFn::Memcpy,
        ir::LibCall::Memset => RuntimeFn::Memset,
        ir::LibCall::Memcmp => RuntimeFn::Memcmp,
        other => bail!("unimplemented libcall {other} (float libcalls should not arise: wasm has direct ops)"),
    })
}

/// Build a runtime helper directly as waffle IR. The signature comes from the call
/// site / declaration (pointer-ish params are wasm i32).
pub(crate) fn synth_runtime_body(
    kind: RuntimeFn,
    module: &waffle::Module,
    sig: waffle::Signature,
    mem0: waffle::Memory,
) -> Result<FunctionBody> {
    let sigdata = &module.signatures[sig];
    // The memory intrinsics shims: (mem, delta) -> prev_pages / (mem) -> pages.
    // The mem index param is carried for signature parity with the LLVM
    // intrinsic and ignored (single linear memory).
    match kind {
        RuntimeFn::MemoryGrow => {
            ensure!(
                sigdata.params.len() == 2 && sigdata.returns.len() == 1,
                "MemoryGrow expects (i32, i32) -> i32, got {sigdata:?}"
            );
            let mut body = FunctionBody::new(module, sig);
            let entry = body.entry;
            let delta = body.blocks[entry].params[1].1;
            let prev =
                body.add_op(entry, Operator::MemoryGrow { mem: mem0 }, &[delta], &[Type::I32]);
            body.set_terminator(entry, Terminator::Return { values: vec![prev] });
            return Ok(body);
        }
        RuntimeFn::MemorySize => {
            ensure!(
                sigdata.params.len() == 1 && sigdata.returns.len() == 1,
                "MemorySize expects (i32) -> i32, got {sigdata:?}"
            );
            let mut body = FunctionBody::new(module, sig);
            let entry = body.entry;
            let pages = body.add_op(entry, Operator::MemorySize { mem: mem0 }, &[], &[Type::I32]);
            body.set_terminator(entry, Terminator::Return { values: vec![pages] });
            return Ok(body);
        }
        _ => {}
    }
    ensure!(
        sigdata.params.len() == 3 && sigdata.params.iter().all(|&t| t == Type::I32),
        "runtime {kind:?} expects (i32, i32, i32) params, got {:?}",
        sigdata.params
    );
    let returns_ptr = !sigdata.returns.is_empty();
    let mut body = FunctionBody::new(module, sig);
    let entry = body.entry;
    let a = body.blocks[entry].params[0].1;
    let b = body.blocks[entry].params[1].1;
    let n = body.blocks[entry].params[2].1;
    match kind {
        RuntimeFn::Memcpy | RuntimeFn::Memmove => {
            // memory.copy is overlap-safe, so one body serves both names.
            body.add_op(
                entry,
                Operator::MemoryCopy { dst_mem: mem0, src_mem: mem0 },
                &[a, b, n],
                &[],
            );
            let values = if returns_ptr { vec![a] } else { vec![] };
            body.set_terminator(entry, Terminator::Return { values });
        }
        RuntimeFn::Memset => {
            body.add_op(entry, Operator::MemoryFill { mem: mem0 }, &[a, b, n], &[]);
            let values = if returns_ptr { vec![a] } else { vec![] };
            body.set_terminator(entry, Terminator::Return { values });
        }
        // i128 runtime fns are CLIF-synthesized in runtime_i128 and never reach here.
        RuntimeFn::Udivti3
        | RuntimeFn::Divti3
        | RuntimeFn::Umodti3
        | RuntimeFn::Modti3
        | RuntimeFn::U128Mulo
        | RuntimeFn::I128Mulo => bail!("{kind:?} is CLIF-synthesized, not waffle-synthesized"),
        // Handled by the early returns above.
        RuntimeFn::MemoryGrow | RuntimeFn::MemorySize => unreachable!(),
        RuntimeFn::Memcmp => {
            ensure!(returns_ptr, "memcmp must return i32");
            // entry(a,b,n) -> header(i); header: i==n ? ret0 : cmp;
            // cmp: load bytes, equal ? next(i+1 via header) : ret diff.
            let header = body.add_block();
            let i = body.add_blockparam(header, Type::I32);
            let cmp = body.add_block();
            let ret_eq = body.add_block();
            let ret_diff = body.add_block();

            let zero = body.add_op(entry, Operator::I32Const { value: 0 }, &[], &[Type::I32]);
            body.set_terminator(
                entry,
                Terminator::Br { target: waffle::BlockTarget { block: header, args: vec![zero] } },
            );

            let done = body.add_op(header, Operator::I32Eq, &[i, n], &[Type::I32]);
            body.set_terminator(
                header,
                Terminator::CondBr {
                    cond: done,
                    if_true: waffle::BlockTarget { block: ret_eq, args: vec![] },
                    if_false: waffle::BlockTarget { block: cmp, args: vec![] },
                },
            );

            let mem = waffle::MemoryArg { align: 0, offset: 0, memory: mem0 };
            let pa = body.add_op(cmp, Operator::I32Add, &[a, i], &[Type::I32]);
            let ba = body.add_op(cmp, Operator::I32Load8U { memory: mem }, &[pa], &[Type::I32]);
            let pb = body.add_op(cmp, Operator::I32Add, &[b, i], &[Type::I32]);
            let bb = body.add_op(cmp, Operator::I32Load8U { memory: mem }, &[pb], &[Type::I32]);
            let eq = body.add_op(cmp, Operator::I32Eq, &[ba, bb], &[Type::I32]);
            let one = body.add_op(cmp, Operator::I32Const { value: 1 }, &[], &[Type::I32]);
            let i2 = body.add_op(cmp, Operator::I32Add, &[i, one], &[Type::I32]);
            body.set_terminator(
                cmp,
                Terminator::CondBr {
                    cond: eq,
                    if_true: waffle::BlockTarget { block: header, args: vec![i2] },
                    if_false: waffle::BlockTarget { block: ret_diff, args: vec![] },
                },
            );

            // cmp dominates ret_diff, so ba/bb are usable here.
            let d = body.add_op(ret_diff, Operator::I32Sub, &[ba, bb], &[Type::I32]);
            body.set_terminator(ret_diff, Terminator::Return { values: vec![d] });

            let z = body.add_op(ret_eq, Operator::I32Const { value: 0 }, &[], &[Type::I32]);
            body.set_terminator(ret_eq, Terminator::Return { values: vec![z] });
        }
    }
    Ok(body)
}

pub(crate) fn intern_sig(
    module: &mut waffle::Module,
    cache: &mut HashMap<SignatureData, waffle::Signature>,
    data: SignatureData,
) -> waffle::Signature {
    *cache
        .entry(data.clone())
        .or_insert_with(|| module.signatures.push(data))
}

pub(crate) fn signature_data(sig: &ir::Signature) -> Result<SignatureData> {
    let expand_abi = |p: &ir::AbiParam| -> Result<Vec<Type>> {
        match p.purpose {
            ir::ArgumentPurpose::Normal => {}
            // sret is semantically just an i32 out-pointer parameter on wasm32.
            ir::ArgumentPurpose::StructReturn => {
                ensure!(
                    p.value_type == ir::types::I32,
                    "sret param must be i32 (wasm32 pointer), got {}",
                    p.value_type
                );
            }
            other => bail!("unsupported ArgumentPurpose {other:?}"),
        }
        expand_type(p.value_type)
    };
    let mut params = Vec::new();
    for p in &sig.params {
        params.extend(expand_abi(p)?);
    }
    let mut returns = Vec::new();
    for r in &sig.returns {
        returns.extend(expand_abi(r)?);
    }
    Ok(SignatureData { params, returns })
}
