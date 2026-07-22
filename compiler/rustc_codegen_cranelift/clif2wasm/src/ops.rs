//! CLIF opcode -> waffle `Operator` lowering table.
//!
//! waffle's Operator enum is 1:1 with wasm opcodes, so this is instruction selection,
//! not renaming: some CLIF ops expand to short multi-op sequences (condition
//! normalization, shift-amount width fixups, i128 pair legalization). Anything not in
//! the starter set fails with a clear "unimplemented opcode" error — the gap list is a
//! prototype deliverable.

use anyhow::{bail, ensure, Context, Result};
use cranelift_codegen::ir::{
    self, condcodes::FloatCC, condcodes::IntCC, types, InstructionData, Opcode,
};
use waffle::{FunctionBody, MemoryArg, Operator, Type, ValueDef};

use crate::bridge::FnCx;
use crate::i128_lower::{self, ShiftKind};
use crate::map_type;

/// Normalize a CLIF condition value (i8 icmp result, or any int used as a truth
/// value) to a wasm i32. i8/i16/i32 map straight through; i64 gets eqz/eqz; i128 is
/// nonzero iff (lo | hi) is nonzero.
pub fn coerce_cond(
    cx: &FnCx,
    body: &mut FunctionBody,
    wb: waffle::Block,
    cond: ir::Value,
) -> Result<waffle::Value> {
    if cx.vty(cond) == types::I128 {
        let (lo, hi) = cx.pair(cond)?;
        let or = body.add_op(wb, Operator::I64Or, &[lo, hi], &[Type::I64]);
        let z = body.add_op(wb, Operator::I64Eqz, &[or], &[Type::I32]);
        return Ok(body.add_op(wb, Operator::I32Eqz, &[z], &[Type::I32]));
    }
    let wv = cx.one(cond)?;
    match map_type(cx.vty(cond))? {
        Type::I32 => Ok(wv),
        Type::I64 => {
            let z = body.add_op(wb, Operator::I64Eqz, &[wv], &[Type::I32]);
            Ok(body.add_op(wb, Operator::I32Eqz, &[z], &[Type::I32]))
        }
        other => bail!("branch/select condition has non-integer type {other}"),
    }
}

pub fn lower_inst(
    cx: &mut FnCx,
    body: &mut FunctionBody,
    wb: waffle::Block,
    inst: ir::Inst,
) -> Result<()> {
    let dfg = &cx.clif.dfg;
    match dfg.insts[inst] {
        // --- Constants ---
        InstructionData::UnaryImm { opcode: Opcode::Iconst, imm } => {
            let rty = dfg.value_type(dfg.first_result(inst));
            let bits = imm.bits();
            let (op, wty) = match rty {
                types::I8 => (Operator::I32Const { value: bits as u32 & 0xFF }, Type::I32),
                types::I16 => (Operator::I32Const { value: bits as u32 & 0xFFFF }, Type::I32),
                types::I32 => (Operator::I32Const { value: bits as u32 }, Type::I32),
                types::I64 => (Operator::I64Const { value: bits as u64 }, Type::I64),
                other => bail!("iconst of unsupported type {other}"),
            };
            let wv = body.add_op(wb, op, &[], &[wty]);
            cx.bind1(dfg.first_result(inst), wv);
        }
        InstructionData::UnaryIeee32 { opcode: Opcode::F32const, imm } => {
            let wv = body.add_op(wb, Operator::F32Const { value: imm.bits() }, &[], &[Type::F32]);
            cx.bind1(dfg.first_result(inst), wv);
        }
        InstructionData::UnaryIeee64 { opcode: Opcode::F64const, imm } => {
            let wv = body.add_op(wb, Operator::F64Const { value: imm.bits() }, &[], &[Type::F64]);
            cx.bind1(dfg.first_result(inst), wv);
        }

        // --- Unary int conversions (and isplit) ---
        InstructionData::Unary { opcode, arg } => {
            lower_unary(cx, body, wb, inst, opcode, arg)?;
        }

        // cg_clif sprinkles `nop`s as MIR-comment anchors; they carry nothing.
        InstructionData::NullAry { opcode: Opcode::Nop } => {}

        // bitcast (LoadNoOffset format): same-size reinterpretation.
        InstructionData::LoadNoOffset { opcode: Opcode::Bitcast, arg, .. } => {
            let result = dfg.first_result(inst);
            let (src, dst) = (cx.vty(arg), dfg.value_type(result));
            let v = cx.one(arg)?;
            let wv = match (src, dst) {
                (types::F32, types::I32) => {
                    body.add_op(wb, Operator::I32ReinterpretF32, &[v], &[Type::I32])
                }
                (types::I32, types::F32) => {
                    body.add_op(wb, Operator::F32ReinterpretI32, &[v], &[Type::F32])
                }
                (types::F64, types::I64) => {
                    body.add_op(wb, Operator::I64ReinterpretF64, &[v], &[Type::I64])
                }
                (types::I64, types::F64) => {
                    body.add_op(wb, Operator::F64ReinterpretI64, &[v], &[Type::F64])
                }
                (s, d) if s == d => v,
                (s, d) => bail!("bitcast {s} -> {d} unimplemented"),
            };
            cx.bind1(result, wv);
        }

        // --- Statics: symbol_value / global_value -> linear-memory address ---
        InstructionData::UnaryGlobalValue { opcode: Opcode::SymbolValue | Opcode::GlobalValue, global_value } => {
            use cranelift_codegen::ir::GlobalValueData;
            let result = dfg.first_result(inst);
            ensure!(
                dfg.value_type(result) == types::I32,
                "symbol_value result must be i32 (wasm32 pointer), got {} — 64-bit-target CLIF?",
                dfg.value_type(result)
            );
            let gvd = &cx.clif.global_values[global_value];
            let GlobalValueData::Symbol { name, offset, tls, .. } = gvd else {
                bail!("unimplemented global value form {gvd:?} (only Symbol)");
            };
            ensure!(!tls, "TLS symbols are out of scope");
            // Object mode: the address is the linker's to assign — emit a marker
            // global.get; the object writer splices it into i32.const + reloc.
            if let Some(mk) = cx.m.object_markers {
                let ir::ExternalName::User(r) = name else {
                    bail!("unsupported data symbol name {name:?} in object mode");
                };
                let user = cx
                    .clif
                    .params
                    .user_named_funcs()
                    .get(*r)
                    .context("dangling userextname in object mode")?;
                ensure!(
                    user.namespace == 1,
                    "data symbol has namespace {} (expected 1)",
                    user.namespace
                );
                let off: i64 = (*offset).into();
                let g = mk.data.get(&(user.index, off)).with_context(|| {
                    format!("no marker for data ordinal {} offset {off}", user.index)
                })?;
                let wv =
                    body.add_op(wb, Operator::GlobalGet { global_index: *g }, &[], &[Type::I32]);
                cx.bind1(result, wv);
                return Ok(());
            }
            let base = match name {
                ir::ExternalName::User(r) => {
                    // Side table FIRST: in reader-parsed dumps the `userextnameN` ref
                    // indexes cg_clif's original table, which the text doesn't carry —
                    // cranelift-reader rebuilds its own table from `u0:M` callee tokens,
                    // so the ref can FALSELY resolve to a callee entry. The comment-borne
                    // side table is the only trustworthy identity for parsed input.
                    if let Some(sym) = cx.gv_symbols.get(&global_value.as_u32()) {
                        cx.m.data_addr_for_symbol(sym)?
                    } else if let Some(user) = cx.clif.params.user_named_funcs().get(*r) {
                        // Hand-built path: cg_clif convention, namespace 1 = DataId.
                        ensure!(
                            user.namespace == 1,
                            "data symbol has namespace {} (expected 1)",
                            user.namespace
                        );
                        *cx.m.data_addrs.get(user.index as usize).with_context(|| {
                            format!("data index {} out of range ({} objects)", user.index, cx.m.data_addrs.len())
                        })?
                    } else {
                        bail!(
                            "gv{}: dangling userextname and no side-table entry",
                            global_value.as_u32()
                        )
                    }
                }
                other => bail!("unsupported data symbol name {other:?}"),
            };
            let off: i64 = (*offset).into();
            let addr = (base as i64 + off) as u32;
            let wv = body.add_op(wb, Operator::I32Const { value: addr }, &[], &[Type::I32]);
            cx.bind1(result, wv);
        }

        // --- Integer & float binary arithmetic / bitwise ---
        //
        // Narrow ints (i8/i16) live in I32 with a CANONICAL ZERO-EXTENDED representation:
        // loads use Load8U/16U, iconst masks, and add/sub/mul re-mask their result, so
        // unsigned compares and uextend are direct; signed operations sign-extend first.
        InstructionData::Binary { opcode, args } => {
            let result = dfg.first_result(inst);
            let rty = dfg.value_type(result);

            // i128: pair legalization.
            if rty == types::I128 {
                lower_binary_i128(cx, body, wb, result, opcode, args)?;
                return Ok(());
            }

            let wty = map_type(rty)?;
            let a = cx.one(args[0])?;
            let b = cx.one(args[1])?;

            // Shifts: wasm requires amount type == operand type; CLIF allows a
            // mismatch. Both agree the amount is taken mod bit-width, so a plain
            // wrap/extend preserves semantics. (Rotates share the fixup.)
            let is_shift = matches!(
                opcode,
                Opcode::Ishl | Opcode::Ushr | Opcode::Sshr | Opcode::Rotl | Opcode::Rotr
            );
            // wasm rotates work at 32/64-bit width; a narrow rotate is NOT the i32 op.
            ensure!(
                !(matches!(opcode, Opcode::Rotl | Opcode::Rotr)
                    && matches!(rty, types::I8 | types::I16)),
                "narrow-int rotate ({rty}) unimplemented"
            );
            let b = if is_shift {
                let amt_ty = map_type(cx.vty(args[1]))?;
                match (wty, amt_ty) {
                    (Type::I32, Type::I64) => body.add_op(wb, Operator::I32WrapI64, &[b], &[Type::I32]),
                    (Type::I64, Type::I32) => body.add_op(wb, Operator::I64ExtendI32U, &[b], &[Type::I64]),
                    _ => b,
                }
            } else {
                b
            };

            let op = match (opcode, wty) {
                (Opcode::Iadd, Type::I32) => Operator::I32Add,
                (Opcode::Iadd, Type::I64) => Operator::I64Add,
                (Opcode::Isub, Type::I32) => Operator::I32Sub,
                (Opcode::Isub, Type::I64) => Operator::I64Sub,
                (Opcode::Imul, Type::I32) => Operator::I32Mul,
                (Opcode::Imul, Type::I64) => Operator::I64Mul,
                // wasm div/rem trap on zero (and INT_MIN/-1 for signed) exactly like CLIF.
                (Opcode::Udiv, Type::I32) => Operator::I32DivU,
                (Opcode::Udiv, Type::I64) => Operator::I64DivU,
                (Opcode::Sdiv, Type::I32) => Operator::I32DivS,
                (Opcode::Sdiv, Type::I64) => Operator::I64DivS,
                (Opcode::Urem, Type::I32) => Operator::I32RemU,
                (Opcode::Urem, Type::I64) => Operator::I64RemU,
                (Opcode::Srem, Type::I32) => Operator::I32RemS,
                (Opcode::Srem, Type::I64) => Operator::I64RemS,
                (Opcode::Band, Type::I32) => Operator::I32And,
                (Opcode::Band, Type::I64) => Operator::I64And,
                (Opcode::Bor, Type::I32) => Operator::I32Or,
                (Opcode::Bor, Type::I64) => Operator::I64Or,
                (Opcode::Bxor, Type::I32) => Operator::I32Xor,
                (Opcode::Bxor, Type::I64) => Operator::I64Xor,
                (Opcode::Ishl, Type::I32) => Operator::I32Shl,
                (Opcode::Ishl, Type::I64) => Operator::I64Shl,
                (Opcode::Ushr, Type::I32) => Operator::I32ShrU,
                (Opcode::Ushr, Type::I64) => Operator::I64ShrU,
                (Opcode::Sshr, Type::I32) => Operator::I32ShrS,
                (Opcode::Sshr, Type::I64) => Operator::I64ShrS,
                (Opcode::Rotl, Type::I32) => Operator::I32Rotl,
                (Opcode::Rotl, Type::I64) => Operator::I64Rotl,
                (Opcode::Rotr, Type::I32) => Operator::I32Rotr,
                (Opcode::Rotr, Type::I64) => Operator::I64Rotr,
                (Opcode::Fmin, Type::F32) => Operator::F32Min,
                (Opcode::Fmin, Type::F64) => Operator::F64Min,
                (Opcode::Fmax, Type::F32) => Operator::F32Max,
                (Opcode::Fmax, Type::F64) => Operator::F64Max,
                (Opcode::Fcopysign, Type::F32) => Operator::F32Copysign,
                (Opcode::Fcopysign, Type::F64) => Operator::F64Copysign,
                (Opcode::Fadd, Type::F32) => Operator::F32Add,
                (Opcode::Fadd, Type::F64) => Operator::F64Add,
                (Opcode::Fsub, Type::F32) => Operator::F32Sub,
                (Opcode::Fsub, Type::F64) => Operator::F64Sub,
                (Opcode::Fmul, Type::F32) => Operator::F32Mul,
                (Opcode::Fmul, Type::F64) => Operator::F64Mul,
                (Opcode::Fdiv, Type::F32) => Operator::F32Div,
                (Opcode::Fdiv, Type::F64) => Operator::F64Div,
                (op, ty) => bail!("unimplemented binary opcode {op} on {ty}"),
            };
            let mut wv = body.add_op(wb, op, &[a, b], &[wty]);
            // Re-canonicalize narrow results whose op can dirty high bits.
            if matches!(rty, types::I8 | types::I16)
                && matches!(opcode, Opcode::Iadd | Opcode::Isub | Opcode::Imul)
            {
                wv = mask_narrow(body, wb, wv, rty);
            }
            cx.bind1(result, wv);
        }

        // --- Comparisons ---
        InstructionData::IntCompare { opcode: Opcode::Icmp, args, cond } => {
            let operand_clif_ty = cx.vty(args[0]);
            if operand_clif_ty == types::I128 {
                let a = cx.pair(args[0])?;
                let b = cx.pair(args[1])?;
                let wv = i128_lower::icmp(body, wb, cond, a, b)?;
                cx.bind1(dfg.first_result(inst), wv);
                return Ok(());
            }
            let operand_ty = map_type(operand_clif_ty)?;
            ensure!(
                matches!(operand_ty, Type::I32 | Type::I64),
                "icmp on unsupported operand type {operand_ty}"
            );
            let is64 = operand_ty == Type::I64;
            // Narrow operands: canonical form is zero-extended, so eq/ne/unsigned
            // compares are direct; signed compares sign-extend both sides first.
            let narrow = matches!(operand_clif_ty, types::I8 | types::I16);
            let signed_cc = matches!(
                cond,
                IntCC::SignedLessThan
                    | IntCC::SignedLessThanOrEqual
                    | IntCC::SignedGreaterThan
                    | IntCC::SignedGreaterThanOrEqual
            );
            let (a0, b0) = (cx.one(args[0])?, cx.one(args[1])?);
            let (a0, b0) = if narrow && signed_cc {
                (
                    sign_extend_narrow(body, wb, a0, operand_clif_ty),
                    sign_extend_narrow(body, wb, b0, operand_clif_ty),
                )
            } else {
                (a0, b0)
            };
            let op = match cond {
                IntCC::Equal => if is64 { Operator::I64Eq } else { Operator::I32Eq },
                IntCC::NotEqual => if is64 { Operator::I64Ne } else { Operator::I32Ne },
                IntCC::SignedLessThan => if is64 { Operator::I64LtS } else { Operator::I32LtS },
                IntCC::UnsignedLessThan => if is64 { Operator::I64LtU } else { Operator::I32LtU },
                IntCC::SignedLessThanOrEqual => if is64 { Operator::I64LeS } else { Operator::I32LeS },
                IntCC::UnsignedLessThanOrEqual => if is64 { Operator::I64LeU } else { Operator::I32LeU },
                IntCC::SignedGreaterThan => if is64 { Operator::I64GtS } else { Operator::I32GtS },
                IntCC::UnsignedGreaterThan => if is64 { Operator::I64GtU } else { Operator::I32GtU },
                IntCC::SignedGreaterThanOrEqual => if is64 { Operator::I64GeS } else { Operator::I32GeS },
                IntCC::UnsignedGreaterThanOrEqual => if is64 { Operator::I64GeU } else { Operator::I32GeU },
            };
            // wasm comparisons produce i32; CLIF's i8 result folds to I32 in map_type.
            let wv = body.add_op(wb, op, &[a0, b0], &[Type::I32]);
            cx.bind1(dfg.first_result(inst), wv);
        }
        InstructionData::FloatCompare { opcode: Opcode::Fcmp, args, cond } => {
            let is64 = map_type(cx.vty(args[0]))? == Type::F64;
            let a = cx.one(args[0])?;
            let b = cx.one(args[1])?;
            // Direct wasm comparisons (all "ordered": false on NaN except Ne).
            let direct = |c: FloatCC| -> Option<Operator> {
                Some(match (c, is64) {
                    (FloatCC::Equal, true) => Operator::F64Eq,
                    (FloatCC::Equal, false) => Operator::F32Eq,
                    (FloatCC::NotEqual, true) => Operator::F64Ne, // unordered-or-ne == CLIF NotEqual
                    (FloatCC::NotEqual, false) => Operator::F32Ne,
                    (FloatCC::LessThan, true) => Operator::F64Lt,
                    (FloatCC::LessThan, false) => Operator::F32Lt,
                    (FloatCC::LessThanOrEqual, true) => Operator::F64Le,
                    (FloatCC::LessThanOrEqual, false) => Operator::F32Le,
                    (FloatCC::GreaterThan, true) => Operator::F64Gt,
                    (FloatCC::GreaterThan, false) => Operator::F32Gt,
                    (FloatCC::GreaterThanOrEqual, true) => Operator::F64Ge,
                    (FloatCC::GreaterThanOrEqual, false) => Operator::F32Ge,
                    _ => return None,
                })
            };
            let wv = if let Some(op) = direct(cond) {
                body.add_op(wb, op, &[a, b], &[Type::I32])
            } else {
                // Unordered family: derive from the negation of the ordered dual
                // (wasm ordered compares are false on NaN, so !ge == unordered|lt).
                let negated_dual = |c: FloatCC| -> Result<FloatCC> {
                    Ok(match c {
                        FloatCC::UnorderedOrLessThan => FloatCC::GreaterThanOrEqual,
                        FloatCC::UnorderedOrLessThanOrEqual => FloatCC::GreaterThan,
                        FloatCC::UnorderedOrGreaterThan => FloatCC::LessThanOrEqual,
                        FloatCC::UnorderedOrGreaterThanOrEqual => FloatCC::LessThan,
                        other => bail!("unimplemented fcmp condition {other}"),
                    })
                };
                match cond {
                    FloatCC::Ordered => {
                        // a==a && b==b
                        let eq_op = if is64 { Operator::F64Eq } else { Operator::F32Eq };
                        let aa = body.add_op(wb, eq_op, &[a, a], &[Type::I32]);
                        let bb = body.add_op(wb, eq_op, &[b, b], &[Type::I32]);
                        body.add_op(wb, Operator::I32And, &[aa, bb], &[Type::I32])
                    }
                    FloatCC::Unordered => {
                        let ne_op = if is64 { Operator::F64Ne } else { Operator::F32Ne };
                        let aa = body.add_op(wb, ne_op, &[a, a], &[Type::I32]);
                        let bb = body.add_op(wb, ne_op, &[b, b], &[Type::I32]);
                        body.add_op(wb, Operator::I32Or, &[aa, bb], &[Type::I32])
                    }
                    FloatCC::OrderedNotEqual => {
                        // lt | gt (ordered, excludes NaN and equality)
                        let lt = body.add_op(
                            wb,
                            if is64 { Operator::F64Lt } else { Operator::F32Lt },
                            &[a, b],
                            &[Type::I32],
                        );
                        let gt = body.add_op(
                            wb,
                            if is64 { Operator::F64Gt } else { Operator::F32Gt },
                            &[a, b],
                            &[Type::I32],
                        );
                        body.add_op(wb, Operator::I32Or, &[lt, gt], &[Type::I32])
                    }
                    FloatCC::UnorderedOrEqual => {
                        // !(lt | gt)
                        let lt = body.add_op(
                            wb,
                            if is64 { Operator::F64Lt } else { Operator::F32Lt },
                            &[a, b],
                            &[Type::I32],
                        );
                        let gt = body.add_op(
                            wb,
                            if is64 { Operator::F64Gt } else { Operator::F32Gt },
                            &[a, b],
                            &[Type::I32],
                        );
                        let or = body.add_op(wb, Operator::I32Or, &[lt, gt], &[Type::I32]);
                        body.add_op(wb, Operator::I32Eqz, &[or], &[Type::I32])
                    }
                    other => {
                        let dual = direct(negated_dual(other)?)
                            .expect("negated dual is always a direct comparison");
                        let d = body.add_op(wb, dual, &[a, b], &[Type::I32]);
                        body.add_op(wb, Operator::I32Eqz, &[d], &[Type::I32])
                    }
                }
            };
            cx.bind1(dfg.first_result(inst), wv);
        }

        // --- Select ---
        InstructionData::Ternary { opcode: Opcode::Select, args } => {
            let result = dfg.first_result(inst);
            let rty = dfg.value_type(result);
            let c = coerce_cond(cx, body, wb, args[0])?;
            if rty == types::I128 {
                let (xl, xh) = cx.pair(args[1])?;
                let (yl, yh) = cx.pair(args[2])?;
                let lo = body.add_op(wb, Operator::Select, &[xl, yl, c], &[Type::I64]);
                let hi = body.add_op(wb, Operator::Select, &[xh, yh, c], &[Type::I64]);
                cx.bind2(result, lo, hi);
                return Ok(());
            }
            let wty = map_type(rty)?;
            let x = cx.one(args[1])?;
            let y = cx.one(args[2])?;
            // wasm select returns the FIRST operand when cond != 0 — same as CLIF.
            let wv = body.add_op(wb, Operator::Select, &[x, y, c], &[wty]);
            cx.bind1(result, wv);
        }

        // --- Memory ---
        InstructionData::Load { opcode, arg, flags: _, offset } => {
            let addr = lower_addr(cx, arg)?;
            let result = dfg.first_result(inst);
            let rty = dfg.value_type(result);
            if rty == types::I128 && opcode == Opcode::Load {
                let off: i32 = offset.into();
                ensure!(off >= 0, "negative i128 load offset");
                let m0 = mem_arg(cx, offset)?;
                let m8 = mem_arg(cx, (off + 8).into())?;
                let lo = body.add_op(wb, Operator::I64Load { memory: m0 }, &[addr], &[Type::I64]);
                let hi = body.add_op(wb, Operator::I64Load { memory: m8 }, &[addr], &[Type::I64]);
                cx.bind2(result, lo, hi);
                return Ok(());
            }
            let wty = map_type(rty)?;
            let memory = mem_arg(cx, offset)?;
            let op = match (opcode, rty) {
                // Narrow plain loads produce the canonical zero-extended form.
                (Opcode::Load, types::I8) => Operator::I32Load8U { memory },
                (Opcode::Load, types::I16) => Operator::I32Load16U { memory },
                (Opcode::Load, types::I32) => Operator::I32Load { memory },
                (Opcode::Load, types::I64) => Operator::I64Load { memory },
                (Opcode::Load, types::F32) => Operator::F32Load { memory },
                (Opcode::Load, types::F64) => Operator::F64Load { memory },
                // Explicit widening loads.
                (Opcode::Uload8, types::I32) => Operator::I32Load8U { memory },
                (Opcode::Sload8, types::I32) => Operator::I32Load8S { memory },
                (Opcode::Uload16, types::I32) => Operator::I32Load16U { memory },
                (Opcode::Sload16, types::I32) => Operator::I32Load16S { memory },
                (Opcode::Uload8, types::I64) => Operator::I64Load8U { memory },
                (Opcode::Sload8, types::I64) => Operator::I64Load8S { memory },
                (Opcode::Uload16, types::I64) => Operator::I64Load16U { memory },
                (Opcode::Sload16, types::I64) => Operator::I64Load16S { memory },
                (Opcode::Uload32, types::I64) => Operator::I64Load32U { memory },
                (Opcode::Sload32, types::I64) => Operator::I64Load32S { memory },
                (op, ty) => bail!("unimplemented load `{op}` of type {ty}"),
            };
            let wv = body.add_op(wb, op, &[addr], &[wty]);
            cx.bind1(result, wv);
        }
        InstructionData::Store { opcode, args, flags: _, offset } => {
            // CLIF: store v, p, offset — args are [value, addr]; wasm order is [addr, value].
            if cx.vty(args[0]) == types::I128 && opcode == Opcode::Store {
                let (lo, hi) = cx.pair(args[0])?;
                let addr = lower_addr(cx, args[1])?;
                let off: i32 = offset.into();
                ensure!(off >= 0, "negative i128 store offset");
                let m0 = mem_arg(cx, offset)?;
                let m8 = mem_arg(cx, (off + 8).into())?;
                body.add_op(wb, Operator::I64Store { memory: m0 }, &[addr, lo], &[]);
                body.add_op(wb, Operator::I64Store { memory: m8 }, &[addr, hi], &[]);
                return Ok(());
            }
            let value = cx.one(args[0])?;
            let addr = lower_addr(cx, args[1])?;
            let memory = mem_arg(cx, offset)?;
            let vty = cx.vty(args[0]);
            let op = match (opcode, vty) {
                (Opcode::Store, types::I8) => Operator::I32Store8 { memory },
                (Opcode::Store, types::I16) => Operator::I32Store16 { memory },
                (Opcode::Store, types::I32) => Operator::I32Store { memory },
                (Opcode::Store, types::I64) => Operator::I64Store { memory },
                (Opcode::Store, types::F32) => Operator::F32Store { memory },
                (Opcode::Store, types::F64) => Operator::F64Store { memory },
                // Truncating stores of wider values.
                (Opcode::Istore8, types::I32) => Operator::I32Store8 { memory },
                (Opcode::Istore16, types::I32) => Operator::I32Store16 { memory },
                (Opcode::Istore8, types::I64) => Operator::I64Store8 { memory },
                (Opcode::Istore16, types::I64) => Operator::I64Store16 { memory },
                (Opcode::Istore32, types::I64) => Operator::I64Store32 { memory },
                (op, ty) => bail!("unimplemented store `{op}` of type {ty}"),
            };
            body.add_op(wb, op, &[addr, value], &[]);
        }

        // --- Shadow stack ---
        InstructionData::StackLoad { opcode, stack_slot, offset } => {
            let off: i32 = offset.into();
            match opcode {
                Opcode::StackAddr => {
                    let result = dfg.first_result(inst);
                    ensure!(
                        dfg.value_type(result) == types::I32,
                        "stack_addr result must be i32 (wasm32 pointer width)"
                    );
                    let addr = cx.slot_addr(body, wb, stack_slot, off)?;
                    cx.bind1(result, addr);
                }
                Opcode::StackLoad => {
                    let result = dfg.first_result(inst);
                    let rty = dfg.value_type(result);
                    let addr = cx.slot_addr(body, wb, stack_slot, off)?;
                    if rty == types::I128 {
                        let m0 = mem_arg(cx, 0i32.into())?;
                        let m8 = mem_arg(cx, 8i32.into())?;
                        let lo = body.add_op(wb, Operator::I64Load { memory: m0 }, &[addr], &[Type::I64]);
                        let hi = body.add_op(wb, Operator::I64Load { memory: m8 }, &[addr], &[Type::I64]);
                        cx.bind2(result, lo, hi);
                        return Ok(());
                    }
                    let memory = mem_arg(cx, 0i32.into())?;
                    // Width follows the CLIF type — narrow loads MUST be narrow
                    // (a 4-byte load of an i8 slot reads neighbors).
                    let (op, wty) = match rty {
                        types::I8 => (Operator::I32Load8U { memory }, Type::I32),
                        types::I16 => (Operator::I32Load16U { memory }, Type::I32),
                        types::I32 => (Operator::I32Load { memory }, Type::I32),
                        types::I64 => (Operator::I64Load { memory }, Type::I64),
                        types::F32 => (Operator::F32Load { memory }, Type::F32),
                        types::F64 => (Operator::F64Load { memory }, Type::F64),
                        other => bail!("stack_load of unsupported type {other}"),
                    };
                    let wv = body.add_op(wb, op, &[addr], &[wty]);
                    cx.bind1(result, wv);
                }
                other => bail!("unimplemented stack opcode {other}"),
            }
        }
        InstructionData::StackStore { opcode: Opcode::StackStore, arg, stack_slot, offset } => {
            let off: i32 = offset.into();
            if cx.vty(arg) == types::I128 {
                let (lo, hi) = cx.pair(arg)?;
                let addr = cx.slot_addr(body, wb, stack_slot, off)?;
                let m0 = mem_arg(cx, 0i32.into())?;
                let m8 = mem_arg(cx, 8i32.into())?;
                body.add_op(wb, Operator::I64Store { memory: m0 }, &[addr, lo], &[]);
                body.add_op(wb, Operator::I64Store { memory: m8 }, &[addr, hi], &[]);
                return Ok(());
            }
            let value = cx.one(arg)?;
            let addr = cx.slot_addr(body, wb, stack_slot, off)?;
            let memory = mem_arg(cx, 0i32.into())?;
            // Width follows the CLIF type — a 4-byte store of an i8 CLOBBERS the
            // slot's neighbors (this was the Option<u8> scalar-pair miscompile the
            // C4 differential caught: the tag store wiped the payload byte).
            let op = match cx.vty(arg) {
                types::I8 => Operator::I32Store8 { memory },
                types::I16 => Operator::I32Store16 { memory },
                types::I32 => Operator::I32Store { memory },
                types::I64 => Operator::I64Store { memory },
                types::F32 => Operator::F32Store { memory },
                types::F64 => Operator::F64Store { memory },
                other => bail!("stack_store of unsupported type {other}"),
            };
            body.add_op(wb, op, &[addr, value], &[]);
        }

        // --- Atomics: singlethread target -> plain memory ops (fence = nop) ---
        InstructionData::LoadNoOffset { opcode: Opcode::AtomicLoad, arg, .. } => {
            let addr = lower_addr(cx, arg)?;
            let result = dfg.first_result(inst);
            let wty = map_type(dfg.value_type(result))?;
            let memory = mem_arg(cx, 0i32.into())?;
            let op = match wty {
                Type::I32 => Operator::I32Load { memory },
                Type::I64 => Operator::I64Load { memory },
                other => bail!("atomic_load of {other} unimplemented"),
            };
            let wv = body.add_op(wb, op, &[addr], &[wty]);
            cx.bind1(result, wv);
        }
        InstructionData::StoreNoOffset { opcode: Opcode::AtomicStore, args, .. } => {
            // CLIF atomic_store: args are [value, addr].
            let value = cx.one(args[0])?;
            let addr = lower_addr(cx, args[1])?;
            let memory = mem_arg(cx, 0i32.into())?;
            let op = match map_type(cx.vty(args[0]))? {
                Type::I32 => Operator::I32Store { memory },
                Type::I64 => Operator::I64Store { memory },
                other => bail!("atomic_store of {other} unimplemented"),
            };
            body.add_op(wb, op, &[addr, value], &[]);
        }
        InstructionData::AtomicRmw { opcode: Opcode::AtomicRmw, args, op: rmw_op, .. } => {
            use cranelift_codegen::ir::AtomicRmwOp;
            // args are [addr, operand]; returns the OLD value.
            let addr = lower_addr(cx, args[0])?;
            let operand = cx.one(args[1])?;
            let result = dfg.first_result(inst);
            let wty = map_type(dfg.value_type(result))?;
            let is64 = wty == Type::I64;
            ensure!(matches!(wty, Type::I32 | Type::I64), "atomic_rmw of {wty} unimplemented");
            let memory = mem_arg(cx, 0i32.into())?;
            let (load_op, store_op) = if is64 {
                (Operator::I64Load { memory }, Operator::I64Store { memory })
            } else {
                (Operator::I32Load { memory }, Operator::I32Store { memory })
            };
            let old = body.add_op(wb, load_op, &[addr], &[wty]);
            let pick = |a: Operator, b: Operator| if is64 { b } else { a };
            let new = match rmw_op {
                AtomicRmwOp::Add => body.add_op(wb, pick(Operator::I32Add, Operator::I64Add), &[old, operand], &[wty]),
                AtomicRmwOp::Sub => body.add_op(wb, pick(Operator::I32Sub, Operator::I64Sub), &[old, operand], &[wty]),
                AtomicRmwOp::And => body.add_op(wb, pick(Operator::I32And, Operator::I64And), &[old, operand], &[wty]),
                AtomicRmwOp::Or => body.add_op(wb, pick(Operator::I32Or, Operator::I64Or), &[old, operand], &[wty]),
                AtomicRmwOp::Xor => body.add_op(wb, pick(Operator::I32Xor, Operator::I64Xor), &[old, operand], &[wty]),
                AtomicRmwOp::Xchg => operand,
                AtomicRmwOp::Nand => {
                    let and = body.add_op(wb, pick(Operator::I32And, Operator::I64And), &[old, operand], &[wty]);
                    let ones = if is64 {
                        body.add_op(wb, Operator::I64Const { value: u64::MAX }, &[], &[wty])
                    } else {
                        body.add_op(wb, Operator::I32Const { value: u32::MAX }, &[], &[wty])
                    };
                    body.add_op(wb, pick(Operator::I32Xor, Operator::I64Xor), &[and, ones], &[wty])
                }
                AtomicRmwOp::Umin | AtomicRmwOp::Umax | AtomicRmwOp::Smin | AtomicRmwOp::Smax => {
                    let cmp_op = match (rmw_op, is64) {
                        (AtomicRmwOp::Umin, false) => Operator::I32LtU,
                        (AtomicRmwOp::Umin, true) => Operator::I64LtU,
                        (AtomicRmwOp::Umax, false) => Operator::I32GtU,
                        (AtomicRmwOp::Umax, true) => Operator::I64GtU,
                        (AtomicRmwOp::Smin, false) => Operator::I32LtS,
                        (AtomicRmwOp::Smin, true) => Operator::I64LtS,
                        (AtomicRmwOp::Smax, false) => Operator::I32GtS,
                        (AtomicRmwOp::Smax, true) => Operator::I64GtS,
                        _ => unreachable!(),
                    };
                    let c = body.add_op(wb, cmp_op, &[old, operand], &[Type::I32]);
                    body.add_op(wb, Operator::Select, &[old, operand, c], &[wty])
                }
            };
            body.add_op(wb, store_op, &[addr, new], &[]);
            cx.bind1(result, old);
        }
        InstructionData::AtomicCas { opcode: Opcode::AtomicCas, args, .. } => {
            // args are [addr, expected, replacement]; returns the OLD value; stores
            // replacement iff old == expected. Single-threaded: branchless
            // load/select/store is exact.
            let addr = lower_addr(cx, args[0])?;
            let expected = cx.one(args[1])?;
            let replacement = cx.one(args[2])?;
            let result = dfg.first_result(inst);
            let wty = map_type(dfg.value_type(result))?;
            let is64 = wty == Type::I64;
            ensure!(matches!(wty, Type::I32 | Type::I64), "atomic_cas of {wty} unimplemented");
            let memory = mem_arg(cx, 0i32.into())?;
            let (load_op, store_op, eq_op) = if is64 {
                (Operator::I64Load { memory }, Operator::I64Store { memory }, Operator::I64Eq)
            } else {
                (Operator::I32Load { memory }, Operator::I32Store { memory }, Operator::I32Eq)
            };
            let old = body.add_op(wb, load_op, &[addr], &[wty]);
            let eq = body.add_op(wb, eq_op, &[old, expected], &[Type::I32]);
            let new = body.add_op(wb, Operator::Select, &[replacement, old, eq], &[wty]);
            body.add_op(wb, store_op, &[addr, new], &[]);
            cx.bind1(result, old);
        }
        InstructionData::NullAry { opcode: Opcode::Fence } => {}

        // --- Function pointers ---
        InstructionData::FuncAddr { opcode: Opcode::FuncAddr, func_ref } => {
            let result = dfg.first_result(inst);
            ensure!(
                dfg.value_type(result) == types::I32,
                "func_addr result must be i32 (a table index on wasm32)"
            );
            let (ordinal, _) = resolve_callee(cx, func_ref)?;
            let ordinal =
                ordinal.context("func_addr of a libcall helper is unsupported (no table slot)")?;
            // Object mode: the table slot is the linker's to assign — marker global.
            if let Some(mk) = cx.m.object_markers {
                let g = mk
                    .funcs
                    .get(&ordinal)
                    .with_context(|| format!("no marker for func_addr of ordinal {ordinal}"))?;
                let wv =
                    body.add_op(wb, Operator::GlobalGet { global_index: *g }, &[], &[Type::I32]);
                cx.bind1(result, wv);
                return Ok(());
            }
            // Table slot 0 is null; function k lives at k+1.
            let wv = body.add_op(
                wb,
                Operator::I32Const { value: ordinal + 1 },
                &[],
                &[Type::I32],
            );
            cx.bind1(result, wv);
        }

        // --- Direct calls ---
        InstructionData::Call { opcode: Opcode::Call, args, func_ref } => {
            let (_, callee) = resolve_callee(cx, func_ref)?;
            let mut wargs = Vec::new();
            for &v in args.as_slice(&dfg.value_lists) {
                wargs.extend(cx.expand_use(v)?);
            }
            let op = Operator::Call { function_index: callee };
            bind_call_results(cx, body, wb, inst, op, &wargs)?;
        }

        // --- Indirect calls ---
        InstructionData::CallIndirect { opcode: Opcode::CallIndirect, args, sig_ref } => {
            let all = args.as_slice(&dfg.value_lists);
            ensure!(!all.is_empty(), "call_indirect with no callee operand");
            let callee_ptr = all[0];
            ensure!(
                cx.vty(callee_ptr) == types::I32,
                "call_indirect callee pointer must be i32 (a table index on wasm32)"
            );
            let mut wargs = Vec::new();
            for &v in &all[1..] {
                wargs.extend(cx.expand_use(v)?);
            }
            // wasm call_indirect takes the table index as the LAST operand.
            wargs.push(cx.one(callee_ptr)?);
            let sig_index = *cx
                .sig_map
                .get(&sig_ref)
                .context("call_indirect signature not interned")?;
            let op = Operator::CallIndirect { sig_index, table_index: cx.m.table };
            bind_call_results(cx, body, wb, inst, op, &wargs)?;
        }

        // cranelift 0.130: immediate-form instructions are REAL (in 0.133+ they are
        // builder sugar). Materialize the constant and reuse the binary mappings.
        InstructionData::BinaryImm64 { opcode, arg, imm } => {
            let result = dfg.first_result(inst);
            let rty = dfg.value_type(result);
            ensure!(rty != types::I128, "imm-form on i128 unimplemented ({opcode})");
            let wty = map_type(rty)?;
            let a = cx.one(arg)?;
            let bits = imm.bits();
            let is_shift = matches!(opcode, Opcode::IshlImm | Opcode::UshrImm | Opcode::SshrImm);
            let cbits: u64 = match (wty, rty) {
                (Type::I32, types::I8) if is_shift => (bits as u64) % 8,
                (Type::I32, types::I16) if is_shift => (bits as u64) % 16,
                (Type::I32, types::I8) => bits as u64 & 0xFF,
                (Type::I32, types::I16) => bits as u64 & 0xFFFF,
                (Type::I32, _) => bits as u32 as u64,
                _ => bits as u64,
            };
            let cst = match wty {
                Type::I32 => body.add_op(wb, Operator::I32Const { value: cbits as u32 }, &[], &[Type::I32]),
                Type::I64 => body.add_op(wb, Operator::I64Const { value: cbits }, &[], &[Type::I64]),
                other => bail!("imm-form on {other}"),
            };
            let is64 = wty == Type::I64;
            let pick = |a32: Operator, a64: Operator| if is64 { a64 } else { a32 };
            let (lhs, rhs, op, remask) = match opcode {
                Opcode::IaddImm => (a, cst, pick(Operator::I32Add, Operator::I64Add), true),
                Opcode::ImulImm => (a, cst, pick(Operator::I32Mul, Operator::I64Mul), true),
                Opcode::IrsubImm => (cst, a, pick(Operator::I32Sub, Operator::I64Sub), true),
                Opcode::BandImm => (a, cst, pick(Operator::I32And, Operator::I64And), false),
                Opcode::BorImm => (a, cst, pick(Operator::I32Or, Operator::I64Or), false),
                Opcode::BxorImm => (a, cst, pick(Operator::I32Xor, Operator::I64Xor), false),
                Opcode::IshlImm => (a, cst, pick(Operator::I32Shl, Operator::I64Shl), true),
                Opcode::UshrImm => (a, cst, pick(Operator::I32ShrU, Operator::I64ShrU), false),
                Opcode::SshrImm => {
                    // canonical narrow needs sign-extension before an arithmetic shift
                    let lhs = if matches!(rty, types::I8 | types::I16) {
                        sign_extend_narrow(body, wb, a, rty)
                    } else { a };
                    (lhs, cst, pick(Operator::I32ShrS, Operator::I64ShrS), true)
                }
                Opcode::UdivImm => (a, cst, pick(Operator::I32DivU, Operator::I64DivU), false),
                Opcode::SdivImm => (a, cst, pick(Operator::I32DivS, Operator::I64DivS), true),
                Opcode::UremImm => (a, cst, pick(Operator::I32RemU, Operator::I64RemU), false),
                Opcode::SremImm => (a, cst, pick(Operator::I32RemS, Operator::I64RemS), true),
                other => bail!("unimplemented imm-form opcode {other}"),
            };
            let mut wv = body.add_op(wb, op, &[lhs, rhs], &[wty]);
            if remask && matches!(rty, types::I8 | types::I16) {
                wv = mask_narrow(body, wb, wv, rty);
            }
            cx.bind1(result, wv);
        }
        InstructionData::IntCompareImm { opcode: Opcode::IcmpImm, arg, cond, imm } => {
            let aty = cx.vty(arg);
            let wty = map_type(aty)?;
            let signed_cc = matches!(
                cond,
                IntCC::SignedLessThan | IntCC::SignedLessThanOrEqual
                    | IntCC::SignedGreaterThan | IntCC::SignedGreaterThanOrEqual
            );
            let mut a = cx.one(arg)?;
            let bits = imm.bits();
            let cbits: u64 = match (wty, aty) {
                (Type::I32, types::I8) if !signed_cc => bits as u64 & 0xFF,
                (Type::I32, types::I16) if !signed_cc => bits as u64 & 0xFFFF,
                (Type::I32, _) => bits as u32 as u64,
                _ => bits as u64,
            };
            if signed_cc && matches!(aty, types::I8 | types::I16) {
                a = sign_extend_narrow(body, wb, a, aty);
            }
            let cst = match wty {
                Type::I32 => body.add_op(wb, Operator::I32Const { value: cbits as u32 }, &[], &[Type::I32]),
                Type::I64 => body.add_op(wb, Operator::I64Const { value: cbits }, &[], &[Type::I64]),
                other => bail!("icmp_imm on {other}"),
            };
            let is64 = wty == Type::I64;
            let op = match cond {
                IntCC::Equal => if is64 { Operator::I64Eq } else { Operator::I32Eq },
                IntCC::NotEqual => if is64 { Operator::I64Ne } else { Operator::I32Ne },
                IntCC::SignedLessThan => if is64 { Operator::I64LtS } else { Operator::I32LtS },
                IntCC::UnsignedLessThan => if is64 { Operator::I64LtU } else { Operator::I32LtU },
                IntCC::SignedLessThanOrEqual => if is64 { Operator::I64LeS } else { Operator::I32LeS },
                IntCC::UnsignedLessThanOrEqual => if is64 { Operator::I64LeU } else { Operator::I32LeU },
                IntCC::SignedGreaterThan => if is64 { Operator::I64GtS } else { Operator::I32GtS },
                IntCC::UnsignedGreaterThan => if is64 { Operator::I64GtU } else { Operator::I32GtU },
                IntCC::SignedGreaterThanOrEqual => if is64 { Operator::I64GeS } else { Operator::I32GeS },
                IntCC::UnsignedGreaterThanOrEqual => if is64 { Operator::I64GeU } else { Operator::I32GeU },
            };
            let wv = body.add_op(wb, op, &[a, cst], &[Type::I32]);
            cx.bind1(dfg.first_result(inst), wv);
        }

        ref other => bail!("unimplemented opcode {}", other.opcode()),
    }
    Ok(())
}

fn lower_unary(
    cx: &mut FnCx,
    body: &mut FunctionBody,
    wb: waffle::Block,
    inst: ir::Inst,
    opcode: Opcode,
    arg: ir::Value,
) -> Result<()> {
    let dfg = &cx.clif.dfg;
    let src_ty = cx.vty(arg);
    match opcode {
        // isplit: i128 -> (lo, hi) as two i64 results.
        Opcode::Isplit => {
            ensure!(src_ty == types::I128, "isplit of non-i128 {src_ty}");
            let (lo, hi) = cx.pair(arg)?;
            let results: Vec<ir::Value> = dfg.inst_results(inst).to_vec();
            ensure!(results.len() == 2, "isplit must have 2 results");
            cx.bind1(results[0], lo);
            cx.bind1(results[1], hi);
        }
        Opcode::Uextend | Opcode::Sextend => {
            let result = dfg.first_result(inst);
            let dst_ty = dfg.value_type(result);
            let signed = opcode == Opcode::Sextend;
            let v = cx.one(arg)?;
            // Normalize narrow sources within I32 first: canonical form is already
            // zero-extended, so uextend is a no-op; sextend re-signs in place.
            let (v, src_ty) = match src_ty {
                types::I8 | types::I16 => {
                    let n = if signed { sign_extend_narrow(body, wb, v, src_ty) } else { v };
                    if dst_ty == types::I32 {
                        cx.bind1(result, n);
                        return Ok(());
                    }
                    (n, types::I32)
                }
                types::I32 | types::I64 => (v, src_ty),
                other => bail!("extend from {other} unimplemented"),
            };
            // Widen to i64 if needed.
            let v64 = if src_ty == types::I32 {
                let op = if signed { Operator::I64ExtendI32S } else { Operator::I64ExtendI32U };
                body.add_op(wb, op, &[v], &[Type::I64])
            } else {
                v
            };
            match dst_ty {
                types::I64 => cx.bind1(result, v64),
                types::I128 => {
                    let hi = if signed {
                        let c63 = body.add_op(wb, Operator::I64Const { value: 63 }, &[], &[Type::I64]);
                        body.add_op(wb, Operator::I64ShrS, &[v64, c63], &[Type::I64])
                    } else {
                        body.add_op(wb, Operator::I64Const { value: 0 }, &[], &[Type::I64])
                    };
                    cx.bind2(result, v64, hi);
                }
                other => bail!("extend to {other} unimplemented"),
            }
        }
        // ineg: wasm has no neg — emit 0 - v.
        Opcode::Ineg => {
            let result = dfg.first_result(inst);
            let rty = dfg.value_type(result);
            if rty == types::I128 {
                let zero = body.add_op(wb, Operator::I64Const { value: 0 }, &[], &[Type::I64]);
                let v = cx.pair(arg)?;
                let (lo, hi) = crate::i128_lower::sub(body, wb, (zero, zero), v);
                cx.bind2(result, lo, hi);
                return Ok(());
            }
            let wty = map_type(rty)?;
            let v = cx.one(arg)?;
            let (zero_op, sub_op) = match wty {
                Type::I32 => (Operator::I32Const { value: 0 }, Operator::I32Sub),
                Type::I64 => (Operator::I64Const { value: 0 }, Operator::I64Sub),
                other => bail!("ineg of unsupported type {other}"),
            };
            let zero = body.add_op(wb, zero_op, &[], &[wty]);
            let mut wv = body.add_op(wb, sub_op, &[zero, v], &[wty]);
            if matches!(rty, types::I8 | types::I16) {
                wv = mask_narrow(body, wb, wv, rty);
            }
            cx.bind1(result, wv);
        }
        // bnot: wasm has no not — emit v ^ -1.
        Opcode::Bnot => {
            let result = dfg.first_result(inst);
            let rty = dfg.value_type(result);
            if rty == types::I128 {
                let ones = body.add_op(wb, Operator::I64Const { value: u64::MAX }, &[], &[Type::I64]);
                let v = cx.pair(arg)?;
                let (lo, hi) = crate::i128_lower::logic(body, wb, Operator::I64Xor, v, (ones, ones));
                cx.bind2(result, lo, hi);
                return Ok(());
            }
            let wty = map_type(rty)?;
            let v = cx.one(arg)?;
            let (ones_op, xor_op) = match wty {
                Type::I32 => (Operator::I32Const { value: u32::MAX }, Operator::I32Xor),
                Type::I64 => (Operator::I64Const { value: u64::MAX }, Operator::I64Xor),
                other => bail!("bnot of unsupported type {other}"),
            };
            let ones = body.add_op(wb, ones_op, &[], &[wty]);
            let mut wv = body.add_op(wb, xor_op, &[v, ones], &[wty]);
            if matches!(rty, types::I8 | types::I16) {
                wv = mask_narrow(body, wb, wv, rty);
            }
            cx.bind1(result, wv);
        }
        Opcode::Ireduce => {
            let result = dfg.first_result(inst);
            let dst_ty = dfg.value_type(result);
            let low64 = if src_ty == types::I128 { cx.pair(arg)?.0 } else { cx.one(arg)? };
            let wv = match (src_ty, dst_ty) {
                (types::I128, types::I64) => low64,
                (types::I128, types::I32) | (types::I64, types::I32) => {
                    body.add_op(wb, Operator::I32WrapI64, &[low64], &[Type::I32])
                }
                // Narrow destinations MUST re-canonicalize (mask): the source's high
                // bits are live data, not representation slack. (The battery's
                // cast_chain miscompile is what happens otherwise.)
                (types::I64, types::I8 | types::I16) => {
                    let w = body.add_op(wb, Operator::I32WrapI64, &[low64], &[Type::I32]);
                    mask_narrow(body, wb, w, dst_ty)
                }
                (types::I32, types::I8 | types::I16) => mask_narrow(body, wb, low64, dst_ty),
                (s, d) => bail!("ireduce {s} -> {d} unimplemented"),
            };
            cx.bind1(result, wv);
        }
        // --- Bit-counting ---
        Opcode::Clz | Opcode::Ctz | Opcode::Popcnt => {
            let result = dfg.first_result(inst);
            let rty = dfg.value_type(result);
            if rty == types::I128 {
                let (lo, hi) = cx.pair(arg)?;
                let (lo_r, hi_r) = match opcode {
                    // clz128 = hi==0 ? 64+clz(lo) : clz(hi); ctz mirrors on lo.
                    Opcode::Clz | Opcode::Ctz => {
                        let (primary, secondary, prim_op) = if opcode == Opcode::Clz {
                            (hi, lo, Operator::I64Clz)
                        } else {
                            (lo, hi, Operator::I64Ctz)
                        };
                        let p = body.add_op(wb, prim_op, &[primary], &[Type::I64]);
                        let s = body.add_op(wb, prim_op, &[secondary], &[Type::I64]);
                        let c64 = body.add_op(wb, Operator::I64Const { value: 64 }, &[], &[Type::I64]);
                        let s64 = body.add_op(wb, Operator::I64Add, &[s, c64], &[Type::I64]);
                        let z = body.add_op(wb, Operator::I64Eqz, &[primary], &[Type::I32]);
                        let r = body.add_op(wb, Operator::Select, &[s64, p, z], &[Type::I64]);
                        let zero = body.add_op(wb, Operator::I64Const { value: 0 }, &[], &[Type::I64]);
                        (r, zero)
                    }
                    _ => {
                        let a = body.add_op(wb, Operator::I64Popcnt, &[lo], &[Type::I64]);
                        let b = body.add_op(wb, Operator::I64Popcnt, &[hi], &[Type::I64]);
                        let r = body.add_op(wb, Operator::I64Add, &[a, b], &[Type::I64]);
                        let zero = body.add_op(wb, Operator::I64Const { value: 0 }, &[], &[Type::I64]);
                        (r, zero)
                    }
                };
                cx.bind2(result, lo_r, hi_r);
                return Ok(());
            }
            let wty = map_type(rty)?;
            let v = cx.one(arg)?;
            let op = match (opcode, wty) {
                (Opcode::Clz, Type::I32) => Operator::I32Clz,
                (Opcode::Clz, Type::I64) => Operator::I64Clz,
                (Opcode::Ctz, Type::I32) => Operator::I32Ctz,
                (Opcode::Ctz, Type::I64) => Operator::I64Ctz,
                (Opcode::Popcnt, Type::I32) => Operator::I32Popcnt,
                (Opcode::Popcnt, Type::I64) => Operator::I64Popcnt,
                (o, t) => bail!("{o} on {t} unreachable"),
            };
            let mut wv = body.add_op(wb, op, &[v], &[wty]);
            // Narrow adjustments under canonical zext: clz counts 32-width bits
            // (subtract the pad); ctz of 0 must be the narrow width (OR in the
            // implicit top bit at position `width`); popcnt is already correct.
            if matches!(rty, types::I8 | types::I16) {
                let width = if rty == types::I8 { 8u32 } else { 16 };
                match opcode {
                    Opcode::Clz => {
                        let pad = body.add_op(
                            wb,
                            Operator::I32Const { value: 32 - width },
                            &[],
                            &[Type::I32],
                        );
                        wv = body.add_op(wb, Operator::I32Sub, &[wv, pad], &[Type::I32]);
                    }
                    Opcode::Ctz => {
                        let top = body.add_op(
                            wb,
                            Operator::I32Const { value: 1 << width },
                            &[],
                            &[Type::I32],
                        );
                        let v2 = body.add_op(wb, Operator::I32Or, &[v, top], &[Type::I32]);
                        wv = body.add_op(wb, Operator::I32Ctz, &[v2], &[Type::I32]);
                    }
                    _ => {}
                }
            }
            cx.bind1(result, wv);
        }
        // bitrev: classic mask-and-shift bit reversal (no wasm op).
        Opcode::Bitrev => {
            let result = dfg.first_result(inst);
            let rty = dfg.value_type(result);
            let v_in = cx.one(arg)?;
            let wv = match rty {
                types::I8 | types::I16 | types::I32 => {
                    let r = bitrev32(body, wb, v_in);
                    // Narrow: the reversed bits sit in the TOP of the 32-bit word.
                    match rty {
                        types::I8 => {
                            let c = body.add_op(wb, Operator::I32Const { value: 24 }, &[], &[Type::I32]);
                            body.add_op(wb, Operator::I32ShrU, &[r, c], &[Type::I32])
                        }
                        types::I16 => {
                            let c = body.add_op(wb, Operator::I32Const { value: 16 }, &[], &[Type::I32]);
                            body.add_op(wb, Operator::I32ShrU, &[r, c], &[Type::I32])
                        }
                        _ => r,
                    }
                }
                types::I64 => {
                    // reverse each half with bitrev32, then swap halves.
                    let c32 = body.add_op(wb, Operator::I64Const { value: 32 }, &[], &[Type::I64]);
                    let lo32 = body.add_op(wb, Operator::I32WrapI64, &[v_in], &[Type::I32]);
                    let hi64 = body.add_op(wb, Operator::I64ShrU, &[v_in, c32], &[Type::I64]);
                    let hi32 = body.add_op(wb, Operator::I32WrapI64, &[hi64], &[Type::I32]);
                    let rlo = bitrev32(body, wb, lo32);
                    let rhi = bitrev32(body, wb, hi32);
                    let rlo64 = body.add_op(wb, Operator::I64ExtendI32U, &[rlo], &[Type::I64]);
                    let rhi64 = body.add_op(wb, Operator::I64ExtendI32U, &[rhi], &[Type::I64]);
                    let top = body.add_op(wb, Operator::I64Shl, &[rlo64, c32], &[Type::I64]);
                    body.add_op(wb, Operator::I64Or, &[top, rhi64], &[Type::I64])
                }
                other => bail!("bitrev of {other} unimplemented"),
            };
            cx.bind1(result, wv);
        }

        // bswap: no wasm op — shift/mask expansion.
        Opcode::Bswap => {
            let result = dfg.first_result(inst);
            let rty = dfg.value_type(result);
            let v_in = cx.one(arg)?;
            let wv = match rty {
                types::I16 => {
                    // canonical zext: ((v >> 8) | (v << 8)) & 0xFFFF
                    let c8 = body.add_op(wb, Operator::I32Const { value: 8 }, &[], &[Type::I32]);
                    let hi = body.add_op(wb, Operator::I32ShrU, &[v_in, c8], &[Type::I32]);
                    let lo = body.add_op(wb, Operator::I32Shl, &[v_in, c8], &[Type::I32]);
                    let or = body.add_op(wb, Operator::I32Or, &[hi, lo], &[Type::I32]);
                    mask_narrow(body, wb, or, types::I16)
                }
                types::I32 => bswap32(body, wb, v_in),
                types::I64 => bswap64(body, wb, v_in),
                other => bail!("bswap of {other} unimplemented"),
            };
            cx.bind1(result, wv);
        }

        // --- Float unary ---
        Opcode::Fneg | Opcode::Fabs | Opcode::Sqrt | Opcode::Ceil | Opcode::Floor
        | Opcode::Trunc | Opcode::Nearest => {
            let result = dfg.first_result(inst);
            let wty = map_type(dfg.value_type(result))?;
            let v = cx.one(arg)?;
            let op = match (opcode, wty) {
                (Opcode::Fneg, Type::F32) => Operator::F32Neg,
                (Opcode::Fneg, Type::F64) => Operator::F64Neg,
                (Opcode::Fabs, Type::F32) => Operator::F32Abs,
                (Opcode::Fabs, Type::F64) => Operator::F64Abs,
                (Opcode::Sqrt, Type::F32) => Operator::F32Sqrt,
                (Opcode::Sqrt, Type::F64) => Operator::F64Sqrt,
                (Opcode::Ceil, Type::F32) => Operator::F32Ceil,
                (Opcode::Ceil, Type::F64) => Operator::F64Ceil,
                (Opcode::Floor, Type::F32) => Operator::F32Floor,
                (Opcode::Floor, Type::F64) => Operator::F64Floor,
                (Opcode::Trunc, Type::F32) => Operator::F32Trunc,
                (Opcode::Trunc, Type::F64) => Operator::F64Trunc,
                (Opcode::Nearest, Type::F32) => Operator::F32Nearest,
                (Opcode::Nearest, Type::F64) => Operator::F64Nearest,
                (o, t) => bail!("float unary {o} on {t} unimplemented"),
            };
            let wv = body.add_op(wb, op, &[v], &[wty]);
            cx.bind1(result, wv);
        }

        // --- Float <-> int conversions ---
        Opcode::FcvtToSintSat | Opcode::FcvtToUintSat => {
            let result = dfg.first_result(inst);
            let dst = dfg.value_type(result);
            let signed = opcode == Opcode::FcvtToSintSat;
            let f64src = cx.vty(arg) == types::F64;
            let v = cx.one(arg)?;
            // Wide saturating truncation first (nontrapping-fptoint).
            let wide_dst = if dst == types::I64 { Type::I64 } else { Type::I32 };
            let op = match (wide_dst, f64src, signed) {
                (Type::I32, false, true) => Operator::I32TruncSatF32S,
                (Type::I32, false, false) => Operator::I32TruncSatF32U,
                (Type::I32, true, true) => Operator::I32TruncSatF64S,
                (Type::I32, true, false) => Operator::I32TruncSatF64U,
                (Type::I64, false, true) => Operator::I64TruncSatF32S,
                (Type::I64, false, false) => Operator::I64TruncSatF32U,
                (Type::I64, true, true) => Operator::I64TruncSatF64S,
                (Type::I64, true, false) => Operator::I64TruncSatF64U,
                _ => unreachable!(),
            };
            let mut wv = body.add_op(wb, op, &[v], &[wide_dst]);
            // Narrow destinations saturate at the NARROW range: clamp (with the
            // matching signedness — the unsigned wide result can exceed i32::MAX),
            // then re-canonicalize.
            if matches!(dst, types::I8 | types::I16) {
                if signed {
                    let (lo, hi) = if dst == types::I8 { (-128, 127) } else { (-32768, 32767) };
                    wv = clamp_i32(body, wb, wv, lo, hi);
                } else {
                    let hi = if dst == types::I8 { 255u32 } else { 65535 };
                    let hi_c =
                        body.add_op(wb, Operator::I32Const { value: hi }, &[], &[Type::I32]);
                    let too_hi = body.add_op(wb, Operator::I32GtU, &[wv, hi_c], &[Type::I32]);
                    wv = body.add_op(wb, Operator::Select, &[hi_c, wv, too_hi], &[Type::I32]);
                }
                wv = mask_narrow(body, wb, wv, dst);
            } else if dst == types::I128 {
                bail!("fcvt to i128 unimplemented");
            }
            cx.bind1(result, wv);
        }
        Opcode::FcvtToSint | Opcode::FcvtToUint => {
            // Trapping form: wasm trunc traps on NaN/overflow exactly where CLIF does
            // (i32/i64 destinations only; Rust `as` uses the saturating form anyway).
            let result = dfg.first_result(inst);
            let dst = dfg.value_type(result);
            ensure!(
                matches!(dst, types::I32 | types::I64),
                "trapping fcvt to {dst} unimplemented"
            );
            let signed = opcode == Opcode::FcvtToSint;
            let f64src = cx.vty(arg) == types::F64;
            let v = cx.one(arg)?;
            let (op, wty) = match (dst, f64src, signed) {
                (types::I32, false, true) => (Operator::I32TruncF32S, Type::I32),
                (types::I32, false, false) => (Operator::I32TruncF32U, Type::I32),
                (types::I32, true, true) => (Operator::I32TruncF64S, Type::I32),
                (types::I32, true, false) => (Operator::I32TruncF64U, Type::I32),
                (types::I64, false, true) => (Operator::I64TruncF32S, Type::I64),
                (types::I64, false, false) => (Operator::I64TruncF32U, Type::I64),
                (types::I64, true, true) => (Operator::I64TruncF64S, Type::I64),
                (types::I64, true, false) => (Operator::I64TruncF64U, Type::I64),
                _ => unreachable!(),
            };
            let wv = body.add_op(wb, op, &[v], &[wty]);
            cx.bind1(result, wv);
        }
        Opcode::FcvtFromSint | Opcode::FcvtFromUint => {
            let result = dfg.first_result(inst);
            let dst_f64 = dfg.value_type(result) == types::F64;
            let signed = opcode == Opcode::FcvtFromSint;
            let src_ty = cx.vty(arg);
            ensure!(src_ty != types::I128, "fcvt from i128 unimplemented");
            let mut v = cx.one(arg)?;
            // Narrow sources: canonical zext is exact for unsigned; sign-extend first
            // for signed.
            let src_wide = match src_ty {
                types::I8 | types::I16 => {
                    if signed {
                        v = sign_extend_narrow(body, wb, v, src_ty);
                    }
                    Type::I32
                }
                other => map_type(other)?,
            };
            let (op, wty) = match (dst_f64, src_wide, signed) {
                (false, Type::I32, true) => (Operator::F32ConvertI32S, Type::F32),
                (false, Type::I32, false) => (Operator::F32ConvertI32U, Type::F32),
                (false, Type::I64, true) => (Operator::F32ConvertI64S, Type::F32),
                (false, Type::I64, false) => (Operator::F32ConvertI64U, Type::F32),
                (true, Type::I32, true) => (Operator::F64ConvertI32S, Type::F64),
                (true, Type::I32, false) => (Operator::F64ConvertI32U, Type::F64),
                (true, Type::I64, true) => (Operator::F64ConvertI64S, Type::F64),
                (true, Type::I64, false) => (Operator::F64ConvertI64U, Type::F64),
                (_, t, _) => bail!("fcvt from {t} unimplemented"),
            };
            let wv = body.add_op(wb, op, &[v], &[wty]);
            cx.bind1(result, wv);
        }
        Opcode::Fpromote => {
            let v = cx.one(arg)?;
            let wv = body.add_op(wb, Operator::F64PromoteF32, &[v], &[Type::F64]);
            cx.bind1(dfg.first_result(inst), wv);
        }
        Opcode::Fdemote => {
            let v = cx.one(arg)?;
            let wv = body.add_op(wb, Operator::F32DemoteF64, &[v], &[Type::F32]);
            cx.bind1(dfg.first_result(inst), wv);
        }

        other => bail!("unimplemented unary opcode {other}"),
    }
    Ok(())
}

/// 32-bit bit reversal: swap adjacent bits, pairs, nibbles, then bswap.
fn bitrev32(body: &mut FunctionBody, wb: waffle::Block, v: waffle::Value) -> waffle::Value {
    let mut x = v;
    for (mask, sh) in [(0x5555_5555u32, 1u32), (0x3333_3333, 2), (0x0F0F_0F0F, 4)] {
        let m = body.add_op(wb, Operator::I32Const { value: mask }, &[], &[Type::I32]);
        let s = body.add_op(wb, Operator::I32Const { value: sh }, &[], &[Type::I32]);
        let lo = body.add_op(wb, Operator::I32And, &[x, m], &[Type::I32]);
        let lo_sh = body.add_op(wb, Operator::I32Shl, &[lo, s], &[Type::I32]);
        let hi_shr = body.add_op(wb, Operator::I32ShrU, &[x, s], &[Type::I32]);
        let hi = body.add_op(wb, Operator::I32And, &[hi_shr, m], &[Type::I32]);
        x = body.add_op(wb, Operator::I32Or, &[lo_sh, hi], &[Type::I32]);
    }
    bswap32(body, wb, x)
}

fn bswap32(body: &mut FunctionBody, wb: waffle::Block, v: waffle::Value) -> waffle::Value {
    let c = |body: &mut FunctionBody, k: u32| {
        body.add_op(wb, Operator::I32Const { value: k }, &[], &[Type::I32])
    };
    let c24 = c(body, 24);
    let c8 = c(body, 8);
    let m1 = c(body, 0x0000_FF00);
    let m2 = c(body, 0x00FF_0000);
    let b0 = body.add_op(wb, Operator::I32Shl, &[v, c24], &[Type::I32]);
    let t1 = body.add_op(wb, Operator::I32Shl, &[v, c8], &[Type::I32]);
    let b1 = body.add_op(wb, Operator::I32And, &[t1, m2], &[Type::I32]);
    let t2 = body.add_op(wb, Operator::I32ShrU, &[v, c8], &[Type::I32]);
    let b2 = body.add_op(wb, Operator::I32And, &[t2, m1], &[Type::I32]);
    let b3 = body.add_op(wb, Operator::I32ShrU, &[v, c24], &[Type::I32]);
    let o1 = body.add_op(wb, Operator::I32Or, &[b0, b1], &[Type::I32]);
    let o2 = body.add_op(wb, Operator::I32Or, &[o1, b2], &[Type::I32]);
    body.add_op(wb, Operator::I32Or, &[o2, b3], &[Type::I32])
}

fn bswap64(body: &mut FunctionBody, wb: waffle::Block, v: waffle::Value) -> waffle::Value {
    // Swap halves, then bswap32 each half in i64 form:
    // r = (bswap32(lo) << 32) | bswap32(hi)
    let c32 = body.add_op(wb, Operator::I64Const { value: 32 }, &[], &[Type::I64]);
    let lo32 = body.add_op(wb, Operator::I32WrapI64, &[v], &[Type::I32]);
    let hi64 = body.add_op(wb, Operator::I64ShrU, &[v, c32], &[Type::I64]);
    let hi32 = body.add_op(wb, Operator::I32WrapI64, &[hi64], &[Type::I32]);
    let slo = bswap32(body, wb, lo32);
    let shi = bswap32(body, wb, hi32);
    let slo64 = body.add_op(wb, Operator::I64ExtendI32U, &[slo], &[Type::I64]);
    let shi64 = body.add_op(wb, Operator::I64ExtendI32U, &[shi], &[Type::I64]);
    let hi_part = body.add_op(wb, Operator::I64Shl, &[slo64, c32], &[Type::I64]);
    body.add_op(wb, Operator::I64Or, &[hi_part, shi64], &[Type::I64])
}

/// Clamp an i32 value into [lo, hi] via two selects.
fn clamp_i32(
    body: &mut FunctionBody,
    wb: waffle::Block,
    v: waffle::Value,
    lo: i32,
    hi: i32,
) -> waffle::Value {
    let lo_c = body.add_op(wb, Operator::I32Const { value: lo as u32 }, &[], &[Type::I32]);
    let hi_c = body.add_op(wb, Operator::I32Const { value: hi as u32 }, &[], &[Type::I32]);
    let too_lo = body.add_op(wb, Operator::I32LtS, &[v, lo_c], &[Type::I32]);
    let v1 = body.add_op(wb, Operator::Select, &[lo_c, v, too_lo], &[Type::I32]);
    let too_hi = body.add_op(wb, Operator::I32GtS, &[v1, hi_c], &[Type::I32]);
    body.add_op(wb, Operator::Select, &[hi_c, v1, too_hi], &[Type::I32])
}

fn lower_binary_i128(
    cx: &mut FnCx,
    body: &mut FunctionBody,
    wb: waffle::Block,
    result: ir::Value,
    opcode: Opcode,
    args: [ir::Value; 2],
) -> Result<()> {
    match opcode {
        // iconcat lo, hi -> i128 (first operand is the LOW half).
        Opcode::Iconcat => {
            let lo = cx.one(args[0])?;
            let hi = cx.one(args[1])?;
            cx.bind2(result, lo, hi);
        }
        Opcode::Iadd => {
            let (a, b) = (cx.pair(args[0])?, cx.pair(args[1])?);
            let (lo, hi) = i128_lower::add(body, wb, a, b);
            cx.bind2(result, lo, hi);
        }
        Opcode::Isub => {
            let (a, b) = (cx.pair(args[0])?, cx.pair(args[1])?);
            let (lo, hi) = i128_lower::sub(body, wb, a, b);
            cx.bind2(result, lo, hi);
        }
        Opcode::Imul => {
            let (a, b) = (cx.pair(args[0])?, cx.pair(args[1])?);
            let (lo, hi) = i128_lower::mul(body, wb, a, b);
            cx.bind2(result, lo, hi);
        }
        Opcode::Band | Opcode::Bor | Opcode::Bxor => {
            let op = match opcode {
                Opcode::Band => Operator::I64And,
                Opcode::Bor => Operator::I64Or,
                _ => Operator::I64Xor,
            };
            let (a, b) = (cx.pair(args[0])?, cx.pair(args[1])?);
            let (lo, hi) = i128_lower::logic(body, wb, op, a, b);
            cx.bind2(result, lo, hi);
        }
        Opcode::Ishl | Opcode::Ushr | Opcode::Sshr | Opcode::Rotl | Opcode::Rotr => {
            let v = cx.pair(args[0])?;
            // Coerce the shift amount to i64 (CLIF allows any int width; i128 uses lo).
            let amt_ty = cx.vty(args[1]);
            let amt = if amt_ty == types::I128 {
                cx.pair(args[1])?.0
            } else {
                let raw = cx.one(args[1])?;
                if map_type(amt_ty)? == Type::I32 {
                    body.add_op(wb, Operator::I64ExtendI32U, &[raw], &[Type::I64])
                } else {
                    raw
                }
            };
            if matches!(opcode, Opcode::Rotl | Opcode::Rotr) {
                // rot(v, s) = shl(v, s') | ushr(v, 128 - s') with mod-128 masking on
                // both shifts (s'=0 makes both sides the identity: v | v = v).
                let c128 = body.add_op(wb, Operator::I64Const { value: 128 }, &[], &[Type::I64]);
                let inv = body.add_op(wb, Operator::I64Sub, &[c128, amt], &[Type::I64]);
                let (l_amt, r_amt) =
                    if opcode == Opcode::Rotl { (amt, inv) } else { (inv, amt) };
                let (llo, lhi) = i128_lower::shift(body, wb, ShiftKind::Shl, v, l_amt)?;
                let (rlo, rhi) = i128_lower::shift(body, wb, ShiftKind::UShr, v, r_amt)?;
                let (lo, hi) =
                    i128_lower::logic(body, wb, Operator::I64Or, (llo, lhi), (rlo, rhi));
                cx.bind2(result, lo, hi);
                return Ok(());
            }
            let kind = ShiftKind::from_opcode(opcode)?;
            let (lo, hi) = i128_lower::shift(body, wb, kind, v, amt)?;
            cx.bind2(result, lo, hi);
        }
        other => bail!("unimplemented i128 binary opcode {other} (imul.i128 is a known gap)"),
    }
    Ok(())
}

/// Emit a call-shaped operator and bind its (expanded, possibly multi-value) results.
fn bind_call_results(
    cx: &mut FnCx,
    body: &mut FunctionBody,
    wb: waffle::Block,
    inst: ir::Inst,
    op: Operator,
    wargs: &[waffle::Value],
) -> Result<()> {
    let results: Vec<ir::Value> = cx.clif.dfg.inst_results(inst).to_vec();
    // Flatten result types (i128 -> 2xI64).
    let mut tys = Vec::new();
    let mut widths = Vec::new(); // wasm values per CLIF result
    for &r in &results {
        let expanded = crate::expand_type(cx.clif.dfg.value_type(r))?;
        widths.push(expanded.len());
        tys.extend(expanded);
    }
    let call = body.add_op(wb, op, wargs, &tys);
    match tys.len() {
        0 => {}
        1 => cx.bind1(results[0], call),
        // Multi-value: waffle models the call as one tuple-typed value; each wasm-level
        // result is a PickOutput projection (risk probe: external builders plausibly
        // never exercised this path).
        _ => {
            let mut idx = 0u32;
            for (&r, &w) in results.iter().zip(widths.iter()) {
                let mut picked = Vec::new();
                for _ in 0..w {
                    let ty = tys[idx as usize];
                    let pv = body.add_value(ValueDef::PickOutput(call, idx, ty));
                    body.append_to_block(wb, pv);
                    picked.push(pv);
                    idx += 1;
                }
                match w {
                    1 => cx.bind1(r, picked[0]),
                    2 => cx.bind2(r, picked[0], picked[1]),
                    _ => bail!("unexpected result width {w}"),
                }
            }
        }
    }
    Ok(())
}

/// Re-canonicalize a narrow (i8/i16) value to zero-extended form after an op that
/// can dirty its high bits.
fn mask_narrow(
    body: &mut FunctionBody,
    wb: waffle::Block,
    v: waffle::Value,
    ty: ir::Type,
) -> waffle::Value {
    let mask = if ty == types::I8 { 0xFF } else { 0xFFFF };
    let m = body.add_op(wb, Operator::I32Const { value: mask }, &[], &[Type::I32]);
    body.add_op(wb, Operator::I32And, &[v, m], &[Type::I32])
}

/// Sign-extend a canonical narrow value in place (i8/i16 within I32) via the wasm
/// sign-extension ops.
fn sign_extend_narrow(
    body: &mut FunctionBody,
    wb: waffle::Block,
    v: waffle::Value,
    ty: ir::Type,
) -> waffle::Value {
    let op = if ty == types::I8 { Operator::I32Extend8S } else { Operator::I32Extend16S };
    body.add_op(wb, op, &[v], &[Type::I32])
}

/// The pointer-width tripwire: addresses must be CLIF i32. CLIF captured from a
/// 64-bit target carries i64 addresses; it must fail loudly here, never truncate
/// (see docs/plans/cranelift-wasm-backend.md §8 on silent 32-bit miscompiles).
fn lower_addr(cx: &FnCx, addr: ir::Value) -> Result<waffle::Value> {
    let ty = cx.vty(addr);
    ensure!(
        ty == types::I32,
        "address has CLIF type {ty}, expected i32 — was this CLIF built for a 64-bit pointer target?"
    );
    cx.one(addr)
}

fn mem_arg(cx: &FnCx, offset: ir::immediates::Offset32) -> Result<MemoryArg> {
    let off: i32 = offset.into();
    ensure!(off >= 0, "negative load/store offset {off} is unsupported (wasm offsets are unsigned)");
    Ok(MemoryArg {
        // Byte-alignment hint: always correct, never a trap; tuning comes later.
        align: 0,
        offset: off as u32,
        memory: cx.m.mem0,
    })
}

/// Resolve a CLIF `FuncRef` to (ordinal, waffle `Func`): `ExternalName::User
/// { namespace: 0, index: k }` where `k` is the callee's ordinal in the input slice,
/// or an `ExternalName::LibCall` routed to a synthesized runtime helper (which has no
/// ordinal — its address cannot be taken).
fn resolve_callee(cx: &FnCx, func_ref: ir::FuncRef) -> Result<(Option<u32>, waffle::Func)> {
    let ext = &cx.clif.dfg.ext_funcs[func_ref];
    let name_ref = match ext.name {
        ir::ExternalName::User(r) => r,
        ir::ExternalName::LibCall(lc) => {
            let object_mode = cx.m.object_markers.is_some();
            let kind = match lc {
                ir::LibCall::Memcpy => crate::RuntimeFn::Memcpy,
                // Whole-module mode merges memmove into the overlap-safe
                // memory.copy helper; object mode must keep the real name.
                ir::LibCall::Memmove if object_mode => crate::RuntimeFn::Memmove,
                ir::LibCall::Memmove => crate::RuntimeFn::Memcpy,
                ir::LibCall::Memset => crate::RuntimeFn::Memset,
                ir::LibCall::Memcmp => crate::RuntimeFn::Memcmp,
                other => bail!("unimplemented libcall callee {other}"),
            };
            let f = cx
                .m
                .libcalls
                .get(&kind)
                .copied()
                .with_context(|| format!("libcall {lc} not pre-synthesized"))?;
            return Ok((None, f));
        }
        ref other => bail!("unsupported callee name {other:?} (only ExternalName::User/LibCall)"),
    };
    let user = cx
        .clif
        .params
        .user_named_funcs()
        .get(name_ref)
        .context("dangling UserExternalNameRef")?;
    ensure!(
        user.namespace == 0,
        "callee namespace {} unsupported (convention: namespace 0, index = ordinal)",
        user.namespace
    );
    let f = cx
        .m
        .callees
        .get(user.index as usize)
        .copied()
        .with_context(|| {
            format!("callee index {} out of range ({} functions)", user.index, cx.m.callees.len())
        })?;
    Ok((Some(user.index), f))
}
