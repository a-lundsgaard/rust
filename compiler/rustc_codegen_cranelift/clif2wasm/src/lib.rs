//! clif2wasm — prototype bridge from Cranelift IR (CLIF) to wasm via waffle.
//!
//! Question this prototype answers: can CLIF, built for a 32-bit-pointer target, be
//! translated through waffle's public builder API and backend (validate -> Reducifier ->
//! Ramsey stackify -> treeify -> localify) into valid, correctly-executing wasm?
//! See docs/plans/cranelift-wasm-backend.md §6 ("waffle unproven for a non-wasm frontend").

pub mod bridge;
pub mod i128_lower;
pub mod module;
#[cfg(feature = "module-seam")]
pub mod module_impl;
pub mod object;
pub mod ops;
pub mod runtime_i128;

use std::collections::HashMap;

use anyhow::{bail, Result};
use cranelift_codegen::ir;

/// A CLIF function plus the name it will carry in the emitted wasm module.
pub struct NamedClifFunc<'a> {
    pub name: &'a str,
    pub func: &'a ir::Function,
    /// For reader-parsed dumps only: global-value index -> the symbol identity from the
    /// dump's `gvN = symbol ... ; <comment>` line. Hand-built CLIF resolves data refs
    /// through `user_named_funcs` (namespace 1 = data ordinal) and leaves this empty.
    pub gv_symbols: HashMap<u32, String>,
}

impl<'a> NamedClifFunc<'a> {
    pub fn new(name: &'a str, func: &'a ir::Function) -> Self {
        NamedClifFunc { name, func, gv_symbols: HashMap::new() }
    }

    pub fn with_gv_symbols(mut self, gv_symbols: HashMap<u32, String>) -> Self {
        self.gv_symbols = gv_symbols;
        self
    }
}

/// A static data object to place in linear memory (what cg_clif's `define_data`
/// would hand a real `Module` impl).
pub struct NamedClifData<'a> {
    pub name: &'a str,
    pub bytes: Vec<u8>,
    pub align: u32,
    /// Pointer slots inside the bytes, patched after layout: a data target becomes its
    /// linear-memory address; a function target becomes its funcref-table index (the
    /// same value `func_addr` produces). All slots are 4-byte LE (wasm32 pointers).
    pub relocs: Vec<DataRelocEntry>,
}

impl<'a> NamedClifData<'a> {
    pub fn new(name: &'a str, bytes: Vec<u8>, align: u32) -> Self {
        NamedClifData { name, bytes, align, relocs: vec![] }
    }

    pub fn with_relocs(mut self, relocs: Vec<DataRelocEntry>) -> Self {
        self.relocs = relocs;
        self
    }
}

#[derive(Copy, Clone, Debug)]
pub enum DataRelocTarget {
    /// Ordinal of a function in the input slice (namespace-0 index / FuncId).
    Func(u32),
    /// Ordinal of a data object (namespace-1 index / DataId).
    Data(u32),
}

#[derive(Copy, Clone, Debug)]
pub struct DataRelocEntry {
    pub offset: u32,
    pub target: DataRelocTarget,
    pub addend: i64,
}

/// Runtime functions the translator can synthesize directly as wasm bodies (no CLIF
/// input): the libcall family cg_clif emits calls to. `Memcpy` uses bulk-memory
/// `memory.copy`, which is overlap-safe, so it serves memmove too.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RuntimeFn {
    /// `(dest: i32, src: i32, n: i32) -> dest` via `memory.copy` (covers memmove).
    Memcpy,
    /// Distinct from `Memcpy` ONLY for object mode, where each becomes its own
    /// undefined import (libc memcpy is not overlap-safe, so the names must not
    /// merge). The whole-module synthesized body is the same `memory.copy`.
    Memmove,
    /// `(dest: i32, byte: i32, n: i32) -> dest` via `memory.fill`.
    Memset,
    /// C `memcmp(a, b, n) -> i32` — unsigned byte-wise compare loop.
    Memcmp,
    /// `(u128, u128) -> u128` unsigned division (restoring long division).
    Udivti3,
    /// `(i128, i128) -> i128` signed division (truncating).
    Divti3,
    /// `(u128, u128) -> u128` unsigned remainder.
    Umodti3,
    /// `(i128, i128) -> i128` signed remainder (dividend's sign).
    Modti3,
    /// `(u128, u128, out: i32) -> u128` wrapping product; overflow byte at `*out`.
    U128Mulo,
    /// Signed variant of the above.
    I128Mulo,
    /// `(mem: i32, delta_pages: i32) -> prev_pages` — wasm `memory.grow` (the
    /// fork reroutes `llvm.wasm.memory.grow.i32` here; std's dlmalloc bottoms
    /// out in it).
    MemoryGrow,
    /// `(mem: i32) -> pages` — wasm `memory.size`.
    MemorySize,
}

impl RuntimeFn {
    /// Match a well-known runtime symbol name (the named-libcall path).
    pub fn from_symbol(name: &str) -> Option<RuntimeFn> {
        Some(match name {
            "memcpy" => RuntimeFn::Memcpy,
            "memmove" => RuntimeFn::Memmove,
            "memset" => RuntimeFn::Memset,
            "memcmp" | "bcmp" => RuntimeFn::Memcmp,
            "__udivti3" => RuntimeFn::Udivti3,
            "__divti3" => RuntimeFn::Divti3,
            "__umodti3" => RuntimeFn::Umodti3,
            "__modti3" => RuntimeFn::Modti3,
            "__rust_u128_mulo" => RuntimeFn::U128Mulo,
            "__rust_i128_mulo" => RuntimeFn::I128Mulo,
            "__clif2wasm_memory_grow" => RuntimeFn::MemoryGrow,
            "__clif2wasm_memory_size" => RuntimeFn::MemorySize,
            _ => return None,
        })
    }
}

/// Translate a set of CLIF functions into one wasm module, exporting `export`
/// (which must be the name of one of the functions).
///
/// Direct calls between the functions are resolved by `ExternalName::User { namespace: 0,
/// index: k }` where `k` is the callee's ordinal in `funcs` (cg_clif's FuncId scheme).
pub fn translate_module(funcs: &[NamedClifFunc], export: &str) -> Result<Vec<u8>> {
    module::translate_module(funcs, &[], &[export])
}

/// Like [`translate_module`], with static data objects. Data refs resolve by
/// `ExternalName::User { namespace: 1, index: k }` (cg_clif's DataId scheme) or, for
/// reader-parsed dumps, by each function's `gv_symbols` side table matched against
/// data names.
pub fn translate_module_with_data(
    funcs: &[NamedClifFunc],
    data: &[NamedClifData],
    export: &str,
) -> Result<Vec<u8>> {
    module::translate_module(funcs, data, &[export])
}

/// Like [`translate_module_with_data`], exporting several functions (the Module-seam
/// finish path exports every `Linkage::Export` function).
pub fn translate_module_multi(
    funcs: &[NamedClifFunc],
    data: &[NamedClifData],
    exports: &[&str],
) -> Result<Vec<u8>> {
    module::translate_module(funcs, data, exports)
}

/// Full form: `runtime_overrides` marks slice positions whose bodies are synthesized
/// runtime functions (see [`RuntimeFn`]) — the entry's CLIF contributes only its
/// signature.
pub fn translate_module_with_runtime(
    funcs: &[NamedClifFunc],
    data: &[NamedClifData],
    exports: &[&str],
    runtime_overrides: &HashMap<usize, RuntimeFn>,
) -> Result<Vec<u8>> {
    module::translate_module_impl(funcs, data, exports, runtime_overrides, &HashMap::new())
}

/// Fullest form: `imports` marks slice positions that become REAL wasm imports from
/// the `env` module (name = map value); their CLIF entries contribute only the
/// signature. Imported functions occupy the front of the wasm function index space,
/// as the spec requires.
pub fn translate_module_full(
    funcs: &[NamedClifFunc],
    data: &[NamedClifData],
    exports: &[&str],
    runtime_overrides: &HashMap<usize, RuntimeFn>,
    imports: &HashMap<usize, String>,
) -> Result<Vec<u8>> {
    module::translate_module_impl(funcs, data, exports, runtime_overrides, imports)
}

/// CLIF value type -> wasm value type, for types that map to a SINGLE wasm value.
///
/// i8/i16 fold to I32 — the minimum unavoidable narrow-int handling: `icmp` produces i8,
/// so brif/select conditions are i8-typed. Invariant (documented, not checked): such
/// values are only ever 0/1 or used where high bits cannot matter.
///
/// i128 deliberately fails here: it maps to a PAIR of I64s and every consumer must go
/// through `expand_type` / the pair-aware paths instead.
pub fn map_type(ty: ir::Type) -> Result<waffle::Type> {
    use cranelift_codegen::ir::types;
    Ok(match ty {
        types::I8 | types::I16 | types::I32 => waffle::Type::I32,
        types::I64 => waffle::Type::I64,
        types::F32 => waffle::Type::F32,
        types::F64 => waffle::Type::F64,
        other => bail!("clif2wasm: unsupported CLIF type {other}"),
    })
}

/// CLIF value type -> one or two wasm value types. i128 becomes (lo, hi) as 2×I64 —
/// the plan's ABI decision (§8) applied uniformly to signatures, blockparams, branch
/// args, returns, and call boundaries.
pub fn expand_type(ty: ir::Type) -> Result<Vec<waffle::Type>> {
    if ty == cranelift_codegen::ir::types::I128 {
        Ok(vec![waffle::Type::I64, waffle::Type::I64])
    } else {
        Ok(vec![map_type(ty)?])
    }
}
