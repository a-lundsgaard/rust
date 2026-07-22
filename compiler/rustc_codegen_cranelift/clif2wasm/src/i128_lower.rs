//! i128 legalization: CLIF i128 ops lowered to (lo, hi) pairs of wasm i64 ops.
//!
//! This is Milestone A's "i128 — mandatory" slice (plan §6): carry/borrow synthesis
//! for add/sub, pairwise logic, hi/lo-combined comparisons, and branchless dynamic
//! shifts via `select`. Multiplication is deliberately left on the gap list.

use anyhow::{bail, Result};
use cranelift_codegen::ir::condcodes::IntCC;
use waffle::{FunctionBody, Operator, Type, Value};

type Pair = (Value, Value); // (lo, hi)

fn op1(body: &mut FunctionBody, wb: waffle::Block, op: Operator, args: &[Value], ty: Type) -> Value {
    body.add_op(wb, op, args, &[ty])
}

fn c64(body: &mut FunctionBody, wb: waffle::Block, v: u64) -> Value {
    op1(body, wb, Operator::I64Const { value: v }, &[], Type::I64)
}

/// wasm select: returns `a` when `cond != 0`, else `b`.
fn sel64(body: &mut FunctionBody, wb: waffle::Block, cond: Value, a: Value, b: Value) -> Value {
    op1(body, wb, Operator::Select, &[a, b, cond], Type::I64)
}

pub fn add(body: &mut FunctionBody, wb: waffle::Block, a: Pair, b: Pair) -> Pair {
    let lo = op1(body, wb, Operator::I64Add, &[a.0, b.0], Type::I64);
    let carry_i32 = op1(body, wb, Operator::I64LtU, &[lo, a.0], Type::I32);
    let carry = op1(body, wb, Operator::I64ExtendI32U, &[carry_i32], Type::I64);
    let hi0 = op1(body, wb, Operator::I64Add, &[a.1, b.1], Type::I64);
    let hi = op1(body, wb, Operator::I64Add, &[hi0, carry], Type::I64);
    (lo, hi)
}

pub fn sub(body: &mut FunctionBody, wb: waffle::Block, a: Pair, b: Pair) -> Pair {
    let lo = op1(body, wb, Operator::I64Sub, &[a.0, b.0], Type::I64);
    let borrow_i32 = op1(body, wb, Operator::I64LtU, &[a.0, b.0], Type::I32);
    let borrow = op1(body, wb, Operator::I64ExtendI32U, &[borrow_i32], Type::I64);
    let hi0 = op1(body, wb, Operator::I64Sub, &[a.1, b.1], Type::I64);
    let hi = op1(body, wb, Operator::I64Sub, &[hi0, borrow], Type::I64);
    (lo, hi)
}

/// Full 128-bit multiply from 64-bit halves: lo = lo(a.0*b.0);
/// hi = a.1*b.0 + a.0*b.1 + umulhi64(a.0, b.0), with umulhi64 built from 32-bit
/// partial products.
pub fn mul(body: &mut FunctionBody, wb: waffle::Block, a: Pair, b: Pair) -> Pair {
    use Operator::*;
    let mask32 = c64(body, wb, 0xFFFF_FFFF);
    let c32 = c64(body, wb, 32);
    let lo32 = |body: &mut FunctionBody, v: Value| op1(body, wb, I64And, &[v, mask32], Type::I64);
    let hi32 = |body: &mut FunctionBody, v: Value| op1(body, wb, I64ShrU, &[v, c32], Type::I64);

    let al = lo32(body, a.0);
    let ah = hi32(body, a.0);
    let bl = lo32(body, b.0);
    let bh = hi32(body, b.0);

    // umulhi64(a.0, b.0) with carry-correct 32-bit partials.
    let lo_lo = op1(body, wb, I64Mul, &[al, bl], Type::I64);
    let t1 = op1(body, wb, I64Mul, &[ah, bl], Type::I64);
    let mid1 = {
        let h = hi32(body, lo_lo);
        op1(body, wb, I64Add, &[t1, h], Type::I64)
    };
    let t2 = op1(body, wb, I64Mul, &[al, bh], Type::I64);
    let mid2 = {
        let l = lo32(body, mid1);
        op1(body, wb, I64Add, &[t2, l], Type::I64)
    };
    let t3 = op1(body, wb, I64Mul, &[ah, bh], Type::I64);
    let hi_of_lo = {
        let h1 = hi32(body, mid1);
        let h2 = hi32(body, mid2);
        let s = op1(body, wb, I64Add, &[t3, h1], Type::I64);
        op1(body, wb, I64Add, &[s, h2], Type::I64)
    };

    let lo = op1(body, wb, I64Mul, &[a.0, b.0], Type::I64);
    let cross1 = op1(body, wb, I64Mul, &[a.1, b.0], Type::I64);
    let cross2 = op1(body, wb, I64Mul, &[a.0, b.1], Type::I64);
    let hi = {
        let s = op1(body, wb, I64Add, &[cross1, cross2], Type::I64);
        op1(body, wb, I64Add, &[s, hi_of_lo], Type::I64)
    };
    (lo, hi)
}

pub fn logic(
    body: &mut FunctionBody,
    wb: waffle::Block,
    op: Operator, // an I64 pairwise op: I64And / I64Or / I64Xor
    a: Pair,
    b: Pair,
) -> Pair {
    let lo = op1(body, wb, op, &[a.0, b.0], Type::I64);
    let hi = op1(body, wb, op, &[a.1, b.1], Type::I64);
    (lo, hi)
}

/// i128 comparison -> wasm i32 (0/1). Ordering compares are `strict_hi | (eq_hi & lo_u)`.
pub fn icmp(
    body: &mut FunctionBody,
    wb: waffle::Block,
    cond: IntCC,
    a: Pair,
    b: Pair,
) -> Result<Value> {
    use Operator::*;
    let ordered = |body: &mut FunctionBody, strict_hi_op: Operator, lo_op: Operator| {
        let strict_hi = op1(body, wb, strict_hi_op, &[a.1, b.1], Type::I32);
        let eq_hi = op1(body, wb, I64Eq, &[a.1, b.1], Type::I32);
        let lo_cmp = op1(body, wb, lo_op, &[a.0, b.0], Type::I32);
        let both = op1(body, wb, I32And, &[eq_hi, lo_cmp], Type::I32);
        op1(body, wb, I32Or, &[strict_hi, both], Type::I32)
    };
    Ok(match cond {
        IntCC::Equal => {
            let lo = op1(body, wb, I64Eq, &[a.0, b.0], Type::I32);
            let hi = op1(body, wb, I64Eq, &[a.1, b.1], Type::I32);
            op1(body, wb, I32And, &[lo, hi], Type::I32)
        }
        IntCC::NotEqual => {
            let lo = op1(body, wb, I64Ne, &[a.0, b.0], Type::I32);
            let hi = op1(body, wb, I64Ne, &[a.1, b.1], Type::I32);
            op1(body, wb, I32Or, &[lo, hi], Type::I32)
        }
        IntCC::SignedLessThan => ordered(body, I64LtS, I64LtU),
        IntCC::UnsignedLessThan => ordered(body, I64LtU, I64LtU),
        IntCC::SignedLessThanOrEqual => ordered(body, I64LtS, I64LeU),
        IntCC::UnsignedLessThanOrEqual => ordered(body, I64LtU, I64LeU),
        IntCC::SignedGreaterThan => ordered(body, I64GtS, I64GtU),
        IntCC::UnsignedGreaterThan => ordered(body, I64GtU, I64GtU),
        IntCC::SignedGreaterThanOrEqual => ordered(body, I64GtS, I64GeU),
        IntCC::UnsignedGreaterThanOrEqual => ordered(body, I64GtU, I64GeU),
    })
}

/// Branchless dynamic shifts. `amt` is an i64, taken mod 128 (CLIF semantics).
/// The `s1 == 0` special case exists because `x >> (64 - 0)` re-masks to `x >> 0`
/// in wasm, which would leak the whole word into the spill term.
pub fn shift(
    body: &mut FunctionBody,
    wb: waffle::Block,
    kind: ShiftKind,
    v: Pair,
    amt: Value,
) -> Result<Pair> {
    use Operator::*;
    let c127 = c64(body, wb, 127);
    let s = op1(body, wb, I64And, &[amt, c127], Type::I64);
    let c63 = c64(body, wb, 63);
    let s1 = op1(body, wb, I64And, &[s, c63], Type::I64);
    let c64v = c64(body, wb, 64);
    let big = op1(body, wb, I64GeU, &[s, c64v], Type::I32); // s >= 64
    let inv = op1(body, wb, I64Sub, &[c64v, s1], Type::I64); // 64 - s1 (64 when s1==0)
    let s1_is0 = op1(body, wb, I64Eqz, &[s1], Type::I32);
    let zero = c64(body, wb, 0);

    Ok(match kind {
        ShiftKind::Shl => {
            let spill_raw = op1(body, wb, I64ShrU, &[v.0, inv], Type::I64);
            let spill = sel64(body, wb, s1_is0, zero, spill_raw);
            let hi_shift = op1(body, wb, I64Shl, &[v.1, s1], Type::I64);
            let hi_small = op1(body, wb, I64Or, &[hi_shift, spill], Type::I64);
            let lo_small = op1(body, wb, I64Shl, &[v.0, s1], Type::I64);
            let hi_big = op1(body, wb, I64Shl, &[v.0, s1], Type::I64);
            let hi = sel64(body, wb, big, hi_big, hi_small);
            let lo = sel64(body, wb, big, zero, lo_small);
            (lo, hi)
        }
        ShiftKind::UShr => {
            let spill_raw = op1(body, wb, I64Shl, &[v.1, inv], Type::I64);
            let spill = sel64(body, wb, s1_is0, zero, spill_raw);
            let lo_shift = op1(body, wb, I64ShrU, &[v.0, s1], Type::I64);
            let lo_small = op1(body, wb, I64Or, &[lo_shift, spill], Type::I64);
            let hi_small = op1(body, wb, I64ShrU, &[v.1, s1], Type::I64);
            let lo_big = op1(body, wb, I64ShrU, &[v.1, s1], Type::I64);
            let lo = sel64(body, wb, big, lo_big, lo_small);
            let hi = sel64(body, wb, big, zero, hi_small);
            (lo, hi)
        }
        ShiftKind::SShr => {
            let spill_raw = op1(body, wb, I64Shl, &[v.1, inv], Type::I64);
            let spill = sel64(body, wb, s1_is0, zero, spill_raw);
            let lo_shift = op1(body, wb, I64ShrU, &[v.0, s1], Type::I64);
            let lo_small = op1(body, wb, I64Or, &[lo_shift, spill], Type::I64);
            let hi_small = op1(body, wb, I64ShrS, &[v.1, s1], Type::I64);
            let c63b = c64(body, wb, 63);
            let sign = op1(body, wb, I64ShrS, &[v.1, c63b], Type::I64);
            let lo_big = op1(body, wb, I64ShrS, &[v.1, s1], Type::I64);
            let lo = sel64(body, wb, big, lo_big, lo_small);
            let hi = sel64(body, wb, big, sign, hi_small);
            (lo, hi)
        }
    })
}

#[derive(Copy, Clone)]
pub enum ShiftKind {
    Shl,
    UShr,
    SShr,
}

impl ShiftKind {
    pub fn from_opcode(op: cranelift_codegen::ir::Opcode) -> Result<Self> {
        use cranelift_codegen::ir::Opcode;
        Ok(match op {
            Opcode::Ishl => ShiftKind::Shl,
            Opcode::Ushr => ShiftKind::UShr,
            Opcode::Sshr => ShiftKind::SShr,
            other => bail!("not a shift opcode: {other}"),
        })
    }
}
