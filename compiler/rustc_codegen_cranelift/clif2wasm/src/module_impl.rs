//! S4: the `cranelift_module::Module` implementation — the exact seam cg_clif drives.
//!
//! `WasmModule` sits where `ObjectModule` sits: cg_clif calls `declare_function` /
//! `declare_data` / `define_function` / `define_data` through `&mut dyn Module`, and an
//! inherent `finish()` (finalization is NOT part of the trait — the reason the plan
//! needs a thin cg_clif driver fork) turns the collected CLIF + data into a wasm module
//! via the existing translator.
//!
//! `isa()` returns a **Pulley32** TargetIsa — the only 32-bit-pointer ISA in cranelift
//! 0.133 — so `pointer_type() == I32` and cg_clif's `pointer_ty(tcx) == isa.pointer_type`
//! assertion holds for a wasm32 rustc target with no cg_clif core changes. Nothing ever
//! calls `ctx.compile()`; the ISA is a pointer-width oracle only.

use std::collections::BTreeMap;

use anyhow::anyhow;
use cranelift_codegen::control::ControlPlane;
use cranelift_codegen::ir;
use cranelift_codegen::isa::{self, OwnedTargetIsa, TargetIsa};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_module::{
    DataDescription, DataId, FuncId, Init, Linkage, Module, ModuleDeclarations, ModuleError,
    ModuleReloc, ModuleResult,
};
use target_lexicon::Triple;

use crate::{NamedClifData, NamedClifFunc};

pub struct WasmModule {
    isa: OwnedTargetIsa,
    declarations: ModuleDeclarations,
    funcs: BTreeMap<FuncId, ir::Function>,
    data: BTreeMap<DataId, DataDescription>,
    /// Undefined symbols allowed to become REAL `env` wasm imports (host-provided at
    /// instantiation — the playground's print path). From CLIF2WASM_IMPORTS
    /// (comma-separated) or `allow_import`.
    import_symbols: std::collections::HashSet<String>,
}

impl WasmModule {
    pub fn new() -> anyhow::Result<Self> {
        let mut fb = settings::builder();
        fb.set("opt_level", "none")?;
        let flags = settings::Flags::new(fb);
        let triple: Triple = "pulley32-unknown-unknown".parse().map_err(|e| anyhow!("{e}"))?;
        let isa = isa::lookup(triple)?.finish(flags)?;
        let import_symbols = std::env::var("CLIF2WASM_IMPORTS")
            .map(|v| v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
            .unwrap_or_default();
        Ok(WasmModule {
            isa,
            declarations: ModuleDeclarations::default(),
            funcs: BTreeMap::new(),
            data: BTreeMap::new(),
            import_symbols,
        })
    }

    /// Allow an undefined symbol to become a real `env` wasm import.
    pub fn allow_import(&mut self, name: &str) {
        self.import_symbols.insert(name.to_string());
    }

    /// Finalize: translate every collected function and data object into one wasm
    /// module, exporting every `Linkage::Export` function under its declared name.
    ///
    /// Mirrors `ObjectModule::finish()` in shape (inherent, consumes self) — this is
    /// the method the cg_clif driver fork will call where it now calls
    /// `ObjectProduct` emission.
    pub fn finish(self) -> anyhow::Result<Vec<u8>> {
        // Data objects, in DataId order (= namespace-1 index order).
        let mut data_objs: Vec<NamedClifData> = Vec::new();
        let mut data_names: Vec<String> = Vec::new();
        for (id, decl) in self.declarations.get_data_objects() {
            data_names.push(
                decl.name.clone().unwrap_or_else(|| format!("data{}", id.as_u32())),
            );
        }
        for (i, (id, decl)) in self.declarations.get_data_objects().enumerate() {
            let desc = self.data.get(&id);
            let (bytes, relocs) = match desc {
                Some(d) => {
                    let bytes = match &d.init {
                        Init::Bytes { contents } => contents.to_vec(),
                        Init::Zeros { size } => vec![0; *size],
                        Init::Uninitialized => {
                            anyhow::bail!("data `{}` is uninitialized", data_names[i])
                        }
                    };
                    // Pointer slots inside the object: resolve each reloc's target to
                    // the module-level ordinal; the translator patches actual values
                    // after layout.
                    let lenient =
                        std::env::var_os("CLIF2WASM_LENIENT").is_some_and(|v| v == "1");
                    let mut relocs = Vec::new();
                    for &(offset, fref) in &d.function_relocs {
                        match reloc_target_ordinal(&d.function_decls[fref], 0) {
                            Ok(target) => relocs.push(crate::DataRelocEntry {
                                offset,
                                target: crate::DataRelocTarget::Func(target),
                                addend: 0,
                            }),
                            Err(e) if lenient => eprintln!(
                                "clif2wasm-gap: data reloc skipped in `{}`: {e:#}",
                                data_names[i]
                            ),
                            Err(e) => return Err(anyhow!("data `{}`: {e}", data_names[i])),
                        }
                    }
                    for &(offset, gv, addend) in &d.data_relocs {
                        match reloc_target_ordinal(&d.data_decls[gv], 1) {
                            Ok(target) => relocs.push(crate::DataRelocEntry {
                                offset,
                                target: crate::DataRelocTarget::Data(target),
                                addend,
                            }),
                            Err(e) if lenient => eprintln!(
                                "clif2wasm-gap: data reloc skipped in `{}`: {e:#}",
                                data_names[i]
                            ),
                            Err(e) => return Err(anyhow!("data `{}`: {e}", data_names[i])),
                        }
                    }
                    (bytes, relocs)
                }
                None if std::env::var_os("CLIF2WASM_LENIENT").is_some_and(|v| v == "1") => {
                    // Sysroot tier: cross-crate data import (e.g. core statics seen
                    // from compiler_builtins). Zero placeholder; reads through it are
                    // wrong-but-unreachable rlib-object code, and it's gap-reported.
                    eprintln!(
                        "clif2wasm-gap: data import zero-stubbed: {} ({:?})",
                        data_names[i], decl.linkage
                    );
                    (vec![0u8; 64], Vec::new())
                }
                None => anyhow::bail!(
                    "data `{}` declared ({:?}) but never defined — imports of data are a fork-scope gap",
                    data_names[i],
                    decl.linkage
                ),
            };
            let align = desc.and_then(|d| d.align).unwrap_or(8) as u32;
            data_objs.push(
                NamedClifData::new(&data_names[i], bytes, align).with_relocs(relocs),
            );
        }

        // Functions, in FuncId order (= namespace-0 index order = slice ordinal).
        // Declared-but-undefined functions become `unreachable` stubs (the S3 panic
        // policy applied at the seam; real imports are Milestone-B linker scope).
        let mut fn_names: Vec<String> = Vec::new();
        for (id, decl) in self.declarations.get_functions() {
            fn_names.push(decl.name.clone().unwrap_or_else(|| format!("fn{}", id.as_u32())));
        }
        // Undefined-function policy:
        //   1. well-known runtime symbols (memcpy/memcmp/...) -> synthesized real bodies;
        //   2. panic-family symbols -> `unreachable` trap stubs (correct semantics for
        //      -Cpanic=abort without unwinding);
        //   3. anything else -> hard error naming every symbol — unless
        //      CLIF2WASM_LENIENT=1 (sysroot builds), which stubs them and prints a
        //      `clif2wasm-gap:` line per symbol for the gap ratchet.
        let lenient = std::env::var_os("CLIF2WASM_LENIENT").is_some_and(|v| v == "1");
        let mut stub_storage: Vec<ir::Function> = Vec::new();
        let mut which: Vec<Option<usize>> = Vec::new(); // index into stub_storage for stubs
        let mut runtime_overrides: std::collections::HashMap<usize, crate::RuntimeFn> =
            Default::default();
        let mut import_map: std::collections::HashMap<usize, String> = Default::default();
        let mut missing: Vec<String> = Vec::new();
        for (i, (id, decl)) in self.declarations.get_functions().enumerate() {
            if self.funcs.contains_key(&id) {
                which.push(None);
                continue;
            }
            let name = &fn_names[i];
            let mut f = ir::Function::with_name_signature(
                ir::UserFuncName::user(0, id.as_u32()),
                decl.signature.clone(),
            );
            build_trap_body(&mut f);
            if self.import_symbols.contains(name) {
                import_map.insert(i, name.clone());
            } else if let Some(kind) = crate::RuntimeFn::from_symbol(name) {
                runtime_overrides.insert(i, kind);
            } else if name.contains("panic")
                || name.ends_with("_fail")
                || name.ends_with("_failed")
            {
                // trap stub (diverging error/fail handlers are panic-family)
            } else if lenient {
                eprintln!("clif2wasm-gap: undefined symbol stubbed: {name}");
            } else {
                missing.push(name.clone());
            }
            which.push(Some(stub_storage.len()));
            stub_storage.push(f);
        }
        if !missing.is_empty() {
            anyhow::bail!(
                "undefined non-panic symbols (no linker yet; CLIF2WASM_LENIENT=1 to stub for sysroot builds):\n  {}",
                missing.join("\n  ")
            );
        }
        let mut named: Vec<NamedClifFunc> = Vec::new();
        let mut exports: Vec<&str> = Vec::new();
        for (i, (id, decl)) in self.declarations.get_functions().enumerate() {
            let func: &ir::Function = match which[i] {
                None => &self.funcs[&id],
                Some(s) => &stub_storage[s],
            };
            named.push(NamedClifFunc::new(&fn_names[i], func));
            if decl.linkage == Linkage::Export {
                exports.push(&fn_names[i]);
            }
        }

        crate::translate_module_full(&named, &data_objs, &exports, &runtime_overrides, &import_map)
    }

    /// Finalize as a RELOCATABLE wasm object (Phase B): undefined symbols stay
    /// undefined, data becomes named segments, and the `linking` section carries
    /// the symbol table. No stub/import/lenient policy — resolution is the
    /// linker's job.
    pub fn finish_object(self) -> anyhow::Result<Vec<u8>> {
        use crate::object::{translate_object, ObjBinding, ObjData, ObjDataDef, ObjFunc};

        fn binding_of(linkage: Linkage) -> ObjBinding {
            match linkage {
                Linkage::Local => ObjBinding::Local,
                // Preemptible is where rustc's linkonce_odr monomorphizations land:
                // duplicate copies across objects must merge, i.e. weak binding.
                Linkage::Preemptible => ObjBinding::Weak,
                _ => ObjBinding::Global,
            }
        }

        // Owned name storage (ObjFunc/ObjData borrow).
        let fn_names: Vec<String> = self
            .declarations
            .get_functions()
            .map(|(id, decl)| {
                decl.name.clone().unwrap_or_else(|| format!(".L__unnamed_fn_{}", id.as_u32()))
            })
            .collect();
        let data_names: Vec<String> = self
            .declarations
            .get_data_objects()
            .map(|(id, decl)| {
                decl.name.clone().unwrap_or_else(|| format!(".L__unnamed_{}", id.as_u32()))
            })
            .collect();

        // cg_clif's i128 helper family (__udivti3/__divti3/__umodti3/__modti3/
        // __rust_{u,i}128_mulo) is called with OUR i128-as-2xi64 expansion, which
        // need not match LLVM compiler_builtins' lowering of the same names. Keep
        // the namespaces fully separate: rename our references to
        // __clif2wasm_<name> and DEFINE them (synthesized CLIF) as hidden weak —
        // weak so duplicate copies across our objects merge; LLVM-compiled code
        // keeps resolving the original names to compiler_builtins.
        let mut fn_names = fn_names;
        let mut synth: BTreeMap<usize, crate::RuntimeFn> = BTreeMap::new();
        for (i, (id, _decl)) in self.declarations.get_functions().enumerate() {
            if self.funcs.contains_key(&id) {
                continue;
            }
            let Some(kind) = crate::RuntimeFn::from_symbol(&fn_names[i]) else { continue };
            // Memory libcalls by name (memcpy/...) stay undefined: real libc.
            if matches!(
                kind,
                crate::RuntimeFn::Memcpy
                    | crate::RuntimeFn::Memmove
                    | crate::RuntimeFn::Memset
                    | crate::RuntimeFn::Memcmp
            ) {
                continue;
            }
            if !fn_names[i].starts_with("__clif2wasm_") {
                fn_names[i] = format!(
                    "__clif2wasm_{}",
                    fn_names[i].trim_start_matches("__rust_").trim_start_matches("__")
                );
            }
            synth.insert(i, kind);
        }

        let mut fns: Vec<ObjFunc> = Vec::with_capacity(fn_names.len());
        for (i, (id, decl)) in self.declarations.get_functions().enumerate() {
            use crate::object::ObjDef;
            let (def, binding, hidden) = match (self.funcs.get(&id), synth.get(&i)) {
                (Some(f), _) => (
                    Some(ObjDef::Clif(f)),
                    if decl.name.is_none() {
                        // Anonymous functions have no linkage identity outside
                        // this object.
                        ObjBinding::Local
                    } else {
                        binding_of(decl.linkage)
                    },
                    false,
                ),
                (None, Some(&kind)) => (Some(ObjDef::Runtime(kind)), ObjBinding::Weak, true),
                (None, None) => (
                    None,
                    if decl.name.is_none() {
                        ObjBinding::Local
                    } else {
                        binding_of(decl.linkage)
                    },
                    false,
                ),
            };
            fns.push(ObjFunc {
                name: &fn_names[i],
                binding,
                hidden,
                def,
                sig: decl.signature.clone(),
                gv_symbols: std::collections::HashMap::new(),
            });
        }

        let mut data: Vec<ObjData> = Vec::with_capacity(data_names.len());
        for (i, (id, decl)) in self.declarations.get_data_objects().enumerate() {
            let def = match self.data.get(&id) {
                Some(d) => {
                    let bytes = match &d.init {
                        Init::Bytes { contents } => contents.to_vec(),
                        Init::Zeros { size } => vec![0; *size],
                        Init::Uninitialized => {
                            anyhow::bail!("data `{}` is uninitialized", data_names[i])
                        }
                    };
                    let mut relocs = Vec::new();
                    for &(offset, fref) in &d.function_relocs {
                        relocs.push(crate::DataRelocEntry {
                            offset,
                            target: crate::DataRelocTarget::Func(reloc_target_ordinal(
                                &d.function_decls[fref],
                                0,
                            )?),
                            addend: 0,
                        });
                    }
                    for &(offset, gv, addend) in &d.data_relocs {
                        relocs.push(crate::DataRelocEntry {
                            offset,
                            target: crate::DataRelocTarget::Data(reloc_target_ordinal(
                                &d.data_decls[gv],
                                1,
                            )?),
                            addend,
                        });
                    }
                    Some(ObjDataDef {
                        bytes,
                        align: d.align.unwrap_or(8) as u32,
                        relocs,
                    })
                }
                // Undefined data: a real undefined symbol now — the old
                // "fork-scope gap" bail is object mode's bread and butter.
                None => None,
            };
            data.push(ObjData {
                name: &data_names[i],
                binding: if decl.name.is_none() {
                    ObjBinding::Local
                } else {
                    binding_of(decl.linkage)
                },
                hidden: false,
                def,
            });
        }

        translate_object(&fns, &data)
    }
}

/// Resolve a data reloc's `ModuleRelocTarget` to the ordinal in the expected
/// namespace (0 = FuncId, 1 = DataId); anything else is a named fork-scope gap.
fn reloc_target_ordinal(
    target: &cranelift_module::ModuleRelocTarget,
    expected_ns: u32,
) -> anyhow::Result<u32> {
    match target {
        cranelift_module::ModuleRelocTarget::User { namespace, index } => {
            if *namespace != expected_ns {
                anyhow::bail!("reloc target namespace {namespace} (expected {expected_ns})");
            }
            Ok(*index)
        }
        other => anyhow::bail!("unsupported reloc target {other:?} (libcall/known-symbol)"),
    }
}

/// Hand-rolled `unreachable` body (no cranelift-frontend dep in the library): one
/// block carrying the signature's params, terminated by a trap.
fn build_trap_body(f: &mut ir::Function) {
    let block = f.dfg.make_block();
    f.layout.append_block(block);
    for p in f.signature.params.clone() {
        f.dfg.append_block_param(block, p.value_type);
    }
    let inst = f.dfg.make_inst(ir::InstructionData::Trap {
        opcode: ir::Opcode::Trap,
        code: ir::TrapCode::user(1).unwrap(),
    });
    f.layout.append_inst(inst, block);
}

impl Module for WasmModule {
    fn isa(&self) -> &dyn TargetIsa {
        &*self.isa
    }

    fn declarations(&self) -> &ModuleDeclarations {
        &self.declarations
    }

    fn declare_function(
        &mut self,
        name: &str,
        linkage: Linkage,
        signature: &ir::Signature,
    ) -> ModuleResult<FuncId> {
        let (id, _linkage) = self.declarations.declare_function(name, linkage, signature)?;
        Ok(id)
    }

    fn declare_anonymous_function(&mut self, signature: &ir::Signature) -> ModuleResult<FuncId> {
        self.declarations.declare_anonymous_function(signature)
    }

    fn declare_data(
        &mut self,
        name: &str,
        linkage: Linkage,
        writable: bool,
        tls: bool,
    ) -> ModuleResult<DataId> {
        let (id, _linkage) = self.declarations.declare_data(name, linkage, writable, tls)?;
        Ok(id)
    }

    fn declare_anonymous_data(&mut self, writable: bool, tls: bool) -> ModuleResult<DataId> {
        self.declarations.declare_anonymous_data(writable, tls)
    }

    fn define_function_with_control_plane(
        &mut self,
        func: FuncId,
        ctx: &mut cranelift_codegen::Context,
        _ctrl_plane: &mut ControlPlane,
    ) -> ModuleResult<()> {
        let decl = self.declarations.get_function_decl(func);
        if !decl.linkage.is_definable() {
            return Err(ModuleError::InvalidImportDefinition(
                decl.name.clone().unwrap_or_default(),
            ));
        }
        if self.funcs.contains_key(&func) {
            return Err(ModuleError::DuplicateDefinition(
                decl.name.clone().unwrap_or_default(),
            ));
        }
        // The interception point (plan §5): grab the un-compiled, target-independent
        // CLIF. ctx.compile() is never called.
        self.funcs.insert(func, ctx.func.clone());
        Ok(())
    }

    fn define_function_bytes(
        &mut self,
        _func_id: FuncId,
        _alignment: u64,
        _bytes: &[u8],
        _relocs: &[ModuleReloc],
    ) -> ModuleResult<()> {
        Err(ModuleError::Backend(anyhow!(
            "define_function_bytes: machine-code injection is unsupported by the wasm backend"
        )))
    }

    fn define_data(&mut self, data_id: DataId, data: &DataDescription) -> ModuleResult<()> {
        let decl = self.declarations.get_data_decl(data_id);
        if !decl.linkage.is_definable() {
            return Err(ModuleError::InvalidImportDefinition(
                decl.name.clone().unwrap_or_default(),
            ));
        }
        if self.data.contains_key(&data_id) {
            return Err(ModuleError::DuplicateDefinition(
                decl.name.clone().unwrap_or_default(),
            ));
        }
        self.data.insert(data_id, data.clone());
        Ok(())
    }
}
