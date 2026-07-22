//! Synthesized i128 runtime: the named libcalls cg_clif emits for 128-bit division
//! (`__udivti3`/`__divti3`/`__umodti3`/`__modti3`, sig `(i128, i128) -> i128`) and
//! checked multiplication (`__rust_{u,i}128_mulo`, sig `(i128, i128, out_ptr) ->
//! i128` storing the overflow byte through the pointer).
//!
//! Built as CLIF via cranelift-frontend and fed through our own translator — the
//! i128 pair legalization, loops and selects are all already proven there. Division
//! is bit-by-bit restoring long division (128 iterations): correctness-first; a
//! faster algorithm is a later optimization.

use cranelift_codegen::ir::{
    condcodes::IntCC, types, AbiParam, BlockArg, Function, InstBuilder, MemFlags, Signature,
    UserFuncName, Value,
};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};

use crate::RuntimeFn;

pub fn build(kind: RuntimeFn) -> Option<Function> {
    Some(match kind {
        RuntimeFn::Udivti3 => build_udivmod(false),
        RuntimeFn::Umodti3 => build_udivmod(true),
        RuntimeFn::Divti3 => build_sdivmod(false),
        RuntimeFn::Modti3 => build_sdivmod(true),
        RuntimeFn::U128Mulo => build_mulo(false),
        RuntimeFn::I128Mulo => build_mulo(true),
        _ => return None,
    })
}

fn sig_divmod() -> Signature {
    let mut s = Signature::new(CallConv::SystemV);
    s.params.push(AbiParam::new(types::I128));
    s.params.push(AbiParam::new(types::I128));
    s.returns.push(AbiParam::new(types::I128));
    s
}

fn c128(b: &mut FunctionBuilder, lo: i64, hi: i64) -> Value {
    let l = b.ins().iconst(types::I64, lo);
    let h = b.ins().iconst(types::I64, hi);
    b.ins().iconcat(l, h)
}

/// Emit the restoring-division loop; ends in a block where (q, r) are available and
/// already returned per `want_rem`. Divisor == 0 traps (Rust panics before the call;
/// this is belt-and-braces).
fn build_udivmod(want_rem: bool) -> Function {
    let mut func = Function::with_name_signature(UserFuncName::user(0, 0), sig_divmod());
    let mut fbctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut func, &mut fbctx);
        let entry = b.create_block();
        let start = b.create_block();
        let trap = b.create_block();
        let header = b.create_block();
        let cont = b.create_block();
        let exit = b.create_block();
        b.append_block_params_for_function_params(entry);
        let i_p = b.append_block_param(header, types::I32);
        let q_p = b.append_block_param(header, types::I128);
        let r_p = b.append_block_param(header, types::I128);
        let q_out = b.append_block_param(exit, types::I128);
        let r_out = b.append_block_param(exit, types::I128);

        b.switch_to_block(entry);
        let n = b.block_params(entry)[0];
        let d = b.block_params(entry)[1];
        let zero = c128(&mut b, 0, 0);
        let dz = b.ins().icmp(IntCC::Equal, d, zero);
        b.ins().brif(dz, trap, &[], start, &[]);

        b.switch_to_block(trap);
        b.ins().trap(cranelift_codegen::ir::TrapCode::user(2).unwrap());

        b.switch_to_block(start);
        let i0 = b.ins().iconst(types::I32, 127);
        b.ins().jump(
            header,
            &[BlockArg::Value(i0), BlockArg::Value(zero), BlockArg::Value(zero)],
        );

        b.switch_to_block(header);
        let one128 = c128(&mut b, 1, 0);
        // r2 = (r << 1) | ((n >> i) & 1)
        let sh1 = b.ins().iconst(types::I32, 1);
        let r_shl = b.ins().ishl(r_p, sh1);
        let bit_shifted = b.ins().ushr(n, i_p);
        let bit = b.ins().band(bit_shifted, one128);
        let r2 = b.ins().bor(r_shl, bit);
        // if r2 >= d { r3 = r2 - d; q2 = q | (1 << i) } (branchless)
        let ge = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, r2, d);
        let r_sub = b.ins().isub(r2, d);
        let r3 = b.ins().select(ge, r_sub, r2);
        let qbit = b.ins().ishl(one128, i_p);
        let q_or = b.ins().bor(q_p, qbit);
        let q2 = b.ins().select(ge, q_or, q_p);
        let izero = b.ins().iconst(types::I32, 0);
        let done = b.ins().icmp(IntCC::Equal, i_p, izero);
        b.ins().brif(
            done,
            exit,
            &[BlockArg::Value(q2), BlockArg::Value(r3)],
            cont,
            &[],
        );

        b.switch_to_block(cont);
        let one32 = b.ins().iconst(types::I32, 1);
        let i2 = b.ins().isub(i_p, one32);
        b.ins().jump(
            header,
            &[BlockArg::Value(i2), BlockArg::Value(q2), BlockArg::Value(r3)],
        );

        b.switch_to_block(exit);
        let ret = if want_rem { r_out } else { q_out };
        b.ins().return_(&[ret]);

        b.seal_all_blocks();
        b.finalize();
    }
    func
}

/// abs(v) branchless: s = v >> 127 (all ones if negative); (v ^ s) - s.
fn abs128(b: &mut FunctionBuilder, v: Value) -> (Value, Value) {
    let sh = b.ins().iconst(types::I32, 127);
    let s = b.ins().sshr(v, sh);
    let x = b.ins().bxor(v, s);
    let a = b.ins().isub(x, s);
    (a, s)
}

/// Signed division/remainder in terms of magnitudes + an inner restoring loop.
/// Rust semantics: quotient truncates toward zero; remainder takes the dividend's
/// sign — both match "negate the unsigned result by operand signs".
fn build_sdivmod(want_rem: bool) -> Function {
    let mut func = Function::with_name_signature(UserFuncName::user(0, 0), sig_divmod());
    let mut fbctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut func, &mut fbctx);
        let entry = b.create_block();
        let start = b.create_block();
        let trap = b.create_block();
        let header = b.create_block();
        let cont = b.create_block();
        let exit = b.create_block();
        b.append_block_params_for_function_params(entry);
        let i_p = b.append_block_param(header, types::I32);
        let q_p = b.append_block_param(header, types::I128);
        let r_p = b.append_block_param(header, types::I128);
        let q_out = b.append_block_param(exit, types::I128);
        let r_out = b.append_block_param(exit, types::I128);

        b.switch_to_block(entry);
        let a = b.block_params(entry)[0];
        let dsig = b.block_params(entry)[1];
        let zero = c128(&mut b, 0, 0);
        let dz = b.ins().icmp(IntCC::Equal, dsig, zero);
        b.ins().brif(dz, trap, &[], start, &[]);

        b.switch_to_block(trap);
        b.ins().trap(cranelift_codegen::ir::TrapCode::user(2).unwrap());

        b.switch_to_block(start);
        let (n_abs, sa) = abs128(&mut b, a);
        let (d_abs, sb) = abs128(&mut b, dsig);
        let i0 = b.ins().iconst(types::I32, 127);
        b.ins().jump(
            header,
            &[BlockArg::Value(i0), BlockArg::Value(zero), BlockArg::Value(zero)],
        );

        b.switch_to_block(header);
        let one128 = c128(&mut b, 1, 0);
        let sh1 = b.ins().iconst(types::I32, 1);
        let r_shl = b.ins().ishl(r_p, sh1);
        let bit_shifted = b.ins().ushr(n_abs, i_p);
        let bit = b.ins().band(bit_shifted, one128);
        let r2 = b.ins().bor(r_shl, bit);
        let ge = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, r2, d_abs);
        let r_sub = b.ins().isub(r2, d_abs);
        let r3 = b.ins().select(ge, r_sub, r2);
        let qbit = b.ins().ishl(one128, i_p);
        let q_or = b.ins().bor(q_p, qbit);
        let q2 = b.ins().select(ge, q_or, q_p);
        let izero = b.ins().iconst(types::I32, 0);
        let done = b.ins().icmp(IntCC::Equal, i_p, izero);
        b.ins().brif(
            done,
            exit,
            &[BlockArg::Value(q2), BlockArg::Value(r3)],
            cont,
            &[],
        );

        b.switch_to_block(cont);
        let one32 = b.ins().iconst(types::I32, 1);
        let i2 = b.ins().isub(i_p, one32);
        b.ins().jump(
            header,
            &[BlockArg::Value(i2), BlockArg::Value(q2), BlockArg::Value(r3)],
        );

        b.switch_to_block(exit);
        let ret = if want_rem {
            // remainder sign follows the dividend: r' = (r ^ sa) - sa
            let x = b.ins().bxor(r_out, sa);
            b.ins().isub(x, sa)
        } else {
            // quotient sign = sa ^ sb
            let sq = b.ins().bxor(sa, sb);
            let x = b.ins().bxor(q_out, sq);
            b.ins().isub(x, sq)
        };
        b.ins().return_(&[ret]);

        b.seal_all_blocks();
        b.finalize();
    }
    func
}

/// `__rust_{u,i}128_mulo(a, b, out) -> product` storing the overflow flag byte at
/// `*out`. Unsigned overflow: (both high halves nonzero) OR (cross-term >> 64 != 0)
/// OR carry out of the final add. Signed: work on magnitudes; overflow additionally
/// when the magnitude product exceeds the signed range for the result's sign.
fn build_mulo(signed: bool) -> Function {
    let mut sig = Signature::new(CallConv::SystemV);
    sig.params.push(AbiParam::new(types::I128));
    sig.params.push(AbiParam::new(types::I128));
    sig.params.push(AbiParam::new(types::I32)); // out-pointer (wasm32)
    sig.returns.push(AbiParam::new(types::I128));
    let mut func = Function::with_name_signature(UserFuncName::user(0, 0), sig);
    let mut fbctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut func, &mut fbctx);
        let blk = b.create_block();
        b.append_block_params_for_function_params(blk);
        b.switch_to_block(blk);
        let a_in = b.block_params(blk)[0];
        let b_in = b.block_params(blk)[1];
        let out = b.block_params(blk)[2];

        let (a, b_v, sa, sb) = if signed {
            let (aa, sa) = abs128(&mut b, a_in);
            let (bb, sb) = abs128(&mut b, b_in);
            (aa, bb, Some(sa), Some(sb))
        } else {
            (a_in, b_in, None, None)
        };

        let zero = c128(&mut b, 0, 0);
        // Halves.
        let sh64 = b.ins().iconst(types::I32, 64);
        let a_hi = b.ins().ushr(a, sh64);
        let b_hi = b.ins().ushr(b_v, sh64);
        // both high halves nonzero -> certain overflow
        let ah_nz = b.ins().icmp(IntCC::NotEqual, a_hi, zero);
        let bh_nz = b.ins().icmp(IntCC::NotEqual, b_hi, zero);
        let both_hi = b.ins().band(ah_nz, bh_nz);
        // cross = a_hi*b_lo + a_lo*b_hi; with at most one hi nonzero, one term is 0,
        // so no i128 wrap inside cross itself. Overflow if cross >= 2^64.
        let mask64 = {
            let m = b.ins().iconst(types::I64, -1);
            let z = b.ins().iconst(types::I64, 0);
            b.ins().iconcat(m, z)
        };
        let a_lo = b.ins().band(a, mask64);
        let b_lo = b.ins().band(b_v, mask64);
        let t1 = b.ins().imul(a_hi, b_lo);
        let t2 = b.ins().imul(a_lo, b_hi);
        let cross = b.ins().iadd(t1, t2);
        let cross_hi = b.ins().ushr(cross, sh64);
        let cross_of = b.ins().icmp(IntCC::NotEqual, cross_hi, zero);
        // low product never overflows 128 bits (64x64); final = (cross << 64) + lowprod
        let lowprod = b.ins().imul(a_lo, b_lo);
        let cross_shifted = b.ins().ishl(cross, sh64);
        let mag = b.ins().iadd(cross_shifted, lowprod);
        // carry out of that add: mag < lowprod (unsigned)
        let carry = b.ins().icmp(IntCC::UnsignedLessThan, mag, lowprod);

        let of1 = b.ins().bor(both_hi, cross_of);
        let uoflow = b.ins().bor(of1, carry);

        let (product, oflow) = if signed {
            let (sa, sb) = (sa.unwrap(), sb.unwrap());
            let sq = b.ins().bxor(sa, sb);
            let neg = b.ins().icmp(IntCC::NotEqual, sq, zero);
            // limit: negative results may reach 2^127; positive only 2^127 - 1.
            let lim_hi = b.ins().iconst(types::I64, i64::MIN); // 0x8000...
            let lim_lo = b.ins().iconst(types::I64, 0);
            let bound = b.ins().iconcat(lim_lo, lim_hi); // 2^127 as u128
            let gt = b.ins().icmp(IntCC::UnsignedGreaterThan, mag, bound); // > 2^127
            let ge = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, mag, bound); // >= 2^127
            let range_of = b.ins().select(neg, gt, ge);
            let oflow = b.ins().bor(uoflow, range_of);
            // apply sign: (mag ^ sq) - sq
            let x = b.ins().bxor(mag, sq);
            let signed_prod = b.ins().isub(x, sq);
            (signed_prod, oflow)
        } else {
            (mag, uoflow)
        };

        let of32 = b.ins().uextend(types::I32, oflow);
        b.ins().istore8(MemFlags::new(), of32, out, 0);
        b.ins().return_(&[product]);

        b.seal_all_blocks();
        b.finalize();
    }
    func
}
