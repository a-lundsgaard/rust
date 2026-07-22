//! Per-function translation: CLIF `Function` -> waffle `FunctionBody`.
//!
//! Strategy: two passes, no placeholders. Pass 1 creates every waffle block and
//! blockparam (CLIF blockparams map 1:1, except i128 which becomes a (lo, hi) pair of
//! I64 params). Pass 2 walks blocks in reverse post-order — CLIF is legal SSA, so on
//! the reducible CFGs we accept, every def is translated before any use and a plain
//! value map suffices.
//!
//! Shadow stack: wasm locals have no address, so functions with sized stack slots get
//! a synthesized frame — `sp_new = __stack_pointer - frame_size` in the prologue,
//! `__stack_pointer = sp_old` before every return. `stack_addr` lowers to
//! `sp_new + slot_offset` (the standard LLVM-wasm approach; plan §8).

use std::collections::HashMap;

use anyhow::{bail, ensure, Context, Result};
use cranelift_codegen::dominator_tree::DominatorTree;
use cranelift_codegen::flowgraph::ControlFlowGraph;
use cranelift_codegen::ir::{self, InstructionData};
use waffle::entity::EntityRef;
use waffle::{BlockTarget, FunctionBody, Operator, Terminator, Type};

use crate::{map_type, ops};

/// A CLIF value's waffle representation: one wasm value, or an i128 as (lo, hi) I64s.
#[derive(Copy, Clone, Debug)]
pub enum WVal {
    One(waffle::Value),
    Pair(waffle::Value, waffle::Value),
}

/// Module-level context every function translation needs.
pub struct ModCtx<'a> {
    /// waffle Func index for each input function, by ordinal (call resolution).
    pub callees: &'a [waffle::Func],
    /// The module's single linear memory.
    pub mem0: waffle::Memory,
    /// The synthesized `__stack_pointer` mutable global.
    pub sp_global: waffle::Global,
    /// The single funcref table (index 0 is null; function k lives at k+1).
    pub table: waffle::Table,
    /// Linear-memory address of each data object, by ordinal (namespace-1 index).
    pub data_addrs: &'a [u32],
    /// Data addresses by name, for gv-symbol side-table resolution of reader dumps.
    pub data_by_name: &'a HashMap<String, u32>,
    /// Synthesized runtime helpers for `ExternalName::LibCall` callees.
    pub libcalls: &'a HashMap<crate::RuntimeFn, waffle::Func>,
    /// Object mode (Phase B): symbol references lower to `global.get <marker>`
    /// instead of absolute constants; the object writer's splicer rewrites each
    /// marker site into a padded `i32.const` + relocation. `None` = whole-module
    /// mode (absolute addresses).
    pub object_markers: Option<&'a ObjectMarkers>,
}

/// Marker globals for object-mode symbol references (see `object.rs`).
#[derive(Default)]
pub struct ObjectMarkers {
    /// (data ordinal, byte offset) -> marker global.
    pub data: HashMap<(u32, i64), waffle::Global>,
    /// function ordinal (address-taken) -> marker global.
    pub funcs: HashMap<u32, waffle::Global>,
}

impl ModCtx<'_> {
    /// Resolve a dump-comment symbol string to a data address: exact name match first,
    /// then substring containment (dump comments carry `DefId(.. ~ crate[hash]::NAME)`
    /// or `allocN` forms, not clean linkage names — prototype-grade matching, noted).
    pub fn data_addr_for_symbol(&self, sym: &str) -> Result<u32> {
        if let Some(&a) = self.data_by_name.get(sym) {
            return Ok(a);
        }
        let mut hit = None;
        for (name, &a) in self.data_by_name {
            if sym.contains(name.as_str()) {
                ensure!(hit.is_none(), "gv symbol `{sym}` matches multiple data names");
                hit = Some(a);
            }
        }
        hit.with_context(|| format!("no data object matches gv symbol `{sym}`"))
    }
}

/// Synthesized shadow-stack frame for a function with sized stack slots.
pub struct Frame {
    /// Frame base (`__stack_pointer` after the prologue decrement).
    pub sp_new: waffle::Value,
    /// Caller's stack pointer, restored before every return.
    pub sp_old: waffle::Value,
    /// Byte offset of each CLIF stack slot from `sp_new`.
    pub slots: HashMap<ir::StackSlot, u32>,
}

/// Per-function translation context shared with the operator lowering table.
pub struct FnCx<'a> {
    pub clif: &'a ir::Function,
    pub m: &'a ModCtx<'a>,
    /// CLIF SigRef -> interned waffle signature (for call_indirect).
    pub sig_map: &'a HashMap<ir::SigRef, waffle::Signature>,
    /// gv index -> symbol identity, for reader-parsed dumps (see NamedClifFunc).
    pub gv_symbols: &'a HashMap<u32, String>,
    pub frame: Option<Frame>,
    map: HashMap<ir::Value, WVal>,
}

impl FnCx<'_> {
    fn get(&self, v: ir::Value) -> Result<WVal> {
        let r = self.clif.dfg.resolve_aliases(v);
        self.map.get(&r).copied().with_context(|| {
            format!("no waffle value for CLIF value {r} (use before def — irreducible or unsupported input?)")
        })
    }

    /// CLIF value (alias-resolved) -> single waffle value. Errors on i128 pairs.
    pub fn one(&self, v: ir::Value) -> Result<waffle::Value> {
        match self.get(v)? {
            WVal::One(w) => Ok(w),
            WVal::Pair(..) => bail!("i128 value used where a single wasm value is required"),
        }
    }

    /// CLIF i128 value -> its (lo, hi) pair.
    pub fn pair(&self, v: ir::Value) -> Result<(waffle::Value, waffle::Value)> {
        match self.get(v)? {
            WVal::Pair(lo, hi) => Ok((lo, hi)),
            WVal::One(_) => bail!("non-i128 value used where an i128 pair is required"),
        }
    }

    /// Flatten a CLIF value for ABI positions (branch args, returns, call args).
    pub fn expand_use(&self, v: ir::Value) -> Result<Vec<waffle::Value>> {
        Ok(match self.get(v)? {
            WVal::One(w) => vec![w],
            WVal::Pair(lo, hi) => vec![lo, hi],
        })
    }

    pub fn bind1(&mut self, clif_v: ir::Value, wv: waffle::Value) {
        self.map.insert(self.clif.dfg.resolve_aliases(clif_v), WVal::One(wv));
    }

    pub fn bind2(&mut self, clif_v: ir::Value, lo: waffle::Value, hi: waffle::Value) {
        self.map
            .insert(self.clif.dfg.resolve_aliases(clif_v), WVal::Pair(lo, hi));
    }

    /// CLIF type of an alias-resolved value.
    pub fn vty(&self, v: ir::Value) -> ir::Type {
        self.clif.dfg.value_type(self.clif.dfg.resolve_aliases(v))
    }

    /// Frame-relative address of a stack slot (+ static offset), as a wasm i32.
    pub fn slot_addr(
        &self,
        body: &mut FunctionBody,
        wb: waffle::Block,
        slot: ir::StackSlot,
        offset: i32,
    ) -> Result<waffle::Value> {
        let frame = self.frame.as_ref().context("stack op in a function with no frame")?;
        let base = *frame
            .slots
            .get(&slot)
            .with_context(|| format!("unknown stack slot {slot}"))?;
        ensure!(offset >= 0, "negative stack-slot offset {offset}");
        let total = base + offset as u32;
        let c = body.add_op(wb, Operator::I32Const { value: total }, &[], &[Type::I32]);
        Ok(body.add_op(wb, Operator::I32Add, &[frame.sp_new, c], &[Type::I32]))
    }
}

pub fn translate_function(
    clif: &ir::Function,
    module: &waffle::Module,
    sig: waffle::Signature,
    m: &ModCtx,
    sig_map: &HashMap<ir::SigRef, waffle::Signature>,
    gv_symbols: &HashMap<u32, String>,
) -> Result<FunctionBody> {
    let mut body = FunctionBody::new(module, sig);
    let mut cx = FnCx { clif, m, sig_map, gv_symbols, frame: None, map: HashMap::new() };

    let entry = clif.layout.entry_block().context("function has no entry block")?;

    // Pass 1: blocks and blockparams. The CLIF entry block maps onto the entry block
    // FunctionBody::new pre-created (with params matching the *expanded* signature) —
    // bind CLIF entry params onto them (consuming two per i128) rather than adding new.
    let mut bmap: HashMap<ir::Block, waffle::Block> = HashMap::new();
    {
        let wparams: Vec<waffle::Value> =
            body.blocks[body.entry].params.iter().map(|&(_, v)| v).collect();
        let mut wi = 0;
        for &p in clif.dfg.block_params(entry) {
            if clif.dfg.value_type(p) == ir::types::I128 {
                ensure!(wi + 1 < wparams.len(), "entry params shorter than expanded signature");
                cx.bind2(p, wparams[wi], wparams[wi + 1]);
                wi += 2;
            } else {
                ensure!(wi < wparams.len(), "entry params shorter than expanded signature");
                cx.bind1(p, wparams[wi]);
                wi += 1;
            }
        }
        ensure!(
            wi == wparams.len(),
            "entry block params ({}) do not cover expanded signature params ({})",
            wi,
            wparams.len()
        );
    }
    bmap.insert(entry, body.entry);
    for block in clif.layout.blocks() {
        if block == entry {
            continue;
        }
        let wb = body.add_block();
        bmap.insert(block, wb);
        for &p in clif.dfg.block_params(block) {
            let ty = clif.dfg.value_type(p);
            if ty == ir::types::I128 {
                let lo = body.add_blockparam(wb, Type::I64);
                let hi = body.add_blockparam(wb, Type::I64);
                cx.bind2(p, lo, hi);
            } else {
                let wv = body.add_blockparam(wb, map_type(ty)?);
                cx.bind1(p, wv);
            }
        }
    }

    // Shadow-stack prologue (entry block, before any translated instruction).
    let frame_size = layout_frame(clif, &mut cx)?;
    if frame_size > 0 {
        let wb = body.entry;
        let spg = m.sp_global;
        let sp_old = body.add_op(
            wb,
            Operator::GlobalGet { global_index: spg },
            &[],
            &[Type::I32],
        );
        let c = body.add_op(wb, Operator::I32Const { value: frame_size }, &[], &[Type::I32]);
        let sp_new = body.add_op(wb, Operator::I32Sub, &[sp_old, c], &[Type::I32]);
        body.add_op(wb, Operator::GlobalSet { global_index: spg }, &[sp_new], &[]);
        if let Some(frame) = cx.frame.as_mut() {
            frame.sp_new = sp_new;
            frame.sp_old = sp_old;
        }
    }

    // Pass 2: instructions, in RPO over reachable blocks (cranelift's own analysis).
    let cfg = ControlFlowGraph::with_function(clif);
    let domtree = DominatorTree::with_function(clif, &cfg);
    for &block in domtree.cfg_rpo() {
        let wb = bmap[&block];
        for inst in clif.layout.block_insts(block) {
            match &clif.dfg.insts[inst] {
                InstructionData::Jump { destination, .. } => {
                    let target = block_target(&cx, &bmap, destination)?;
                    body.set_terminator(wb, Terminator::Br { target });
                }
                InstructionData::Brif { arg, blocks, .. } => {
                    let cond = ops::coerce_cond(&cx, &mut body, wb, *arg)?;
                    let if_true = block_target(&cx, &bmap, &blocks[0])?;
                    let if_false = block_target(&cx, &bmap, &blocks[1])?;
                    body.set_terminator(wb, Terminator::CondBr { cond, if_true, if_false });
                }
                InstructionData::BranchTable { arg, table, .. } => {
                    // wasm br_table (waffle Terminator::Select) has CLIF br_table's
                    // exact semantics: index into targets, out-of-range -> default.
                    let jt = &clif.dfg.jump_tables[*table];
                    let default = block_target(&cx, &bmap, &jt.default_block())?;
                    let targets = jt
                        .as_slice()
                        .iter()
                        .map(|bc| block_target(&cx, &bmap, bc))
                        .collect::<Result<Vec<_>>>()?;
                    let raw = cx.one(*arg)?;
                    let value = match crate::map_type(cx.vty(*arg))? {
                        Type::I32 => raw,
                        Type::I64 => body.add_op(wb, Operator::I32WrapI64, &[raw], &[Type::I32]),
                        other => bail!("br_table index has non-integer type {other}"),
                    };
                    body.set_terminator(wb, Terminator::Select { value, targets, default });
                }
                InstructionData::MultiAry { opcode: ir::Opcode::Return, args } => {
                    let mut values = Vec::new();
                    for &v in args.as_slice(&clif.dfg.value_lists) {
                        values.extend(cx.expand_use(v)?);
                    }
                    // Shadow-stack epilogue: restore the caller's stack pointer.
                    if let Some(frame) = cx.frame.as_ref() {
                        body.add_op(
                            wb,
                            Operator::GlobalSet { global_index: m.sp_global },
                            &[frame.sp_old],
                            &[],
                        );
                    }
                    body.set_terminator(wb, Terminator::Return { values });
                }
                InstructionData::Trap { .. } => {
                    body.set_terminator(wb, Terminator::Unreachable);
                }
                _ => {
                    ops::lower_inst(&mut cx, &mut body, wb, inst)
                        .with_context(|| format!("lowering `{}`", clif.dfg.display_inst(inst)))?;
                }
            }
        }
    }

    Ok(body)
}

/// Assign frame offsets to every sized stack slot; returns the (16-aligned) frame size
/// in bytes, 0 if the function has no slots. Prologue values are patched in after.
fn layout_frame(clif: &ir::Function, cx: &mut FnCx) -> Result<u32> {
    if clif.sized_stack_slots.is_empty() {
        return Ok(0);
    }
    let mut slots = HashMap::new();
    let mut cur: u32 = 0;
    for (slot, data) in clif.sized_stack_slots.iter() {
        let align: u32 = 1 << data.align_shift;
        cur = (cur + align - 1) & !(align - 1);
        slots.insert(slot, cur);
        cur += data.size;
    }
    let frame_size = (cur + 15) & !15;
    // sp_new/sp_old are patched in by the prologue right after this returns; use
    // placeholders that would fail loudly if ever read unpatched.
    cx.frame = Some(Frame {
        sp_new: waffle::Value::invalid(),
        sp_old: waffle::Value::invalid(),
        slots,
    });
    Ok(frame_size)
}

fn block_target(
    cx: &FnCx,
    bmap: &HashMap<ir::Block, waffle::Block>,
    bc: &ir::BlockCall,
) -> Result<BlockTarget> {
    let pool = &cx.clif.dfg.value_lists;
    let block = *bmap
        .get(&bc.block(pool))
        .context("branch to block missing from layout")?;
    let mut args = Vec::new();
    for a in bc.args(pool) {
        match a {
            ir::BlockArg::Value(v) => args.extend(cx.expand_use(v)?),
            other => bail!("unsupported block argument {other:?} (try_call is out of scope)"),
        }
    }
    Ok(BlockTarget { block, args })
}
