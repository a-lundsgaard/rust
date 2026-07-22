//! riwl — a minimal wasm static linker (Phase B of the cranelift→wasm backend).
//!
//! Scope is EXACTLY the verified input universe of this project: relocatable wasm
//! objects emitted by clif2wasm and by rustc's LLVM backend (stock wasip1
//! std/core rlibs, wasi-libc's libc.a, crt1-command.o). That universe was scanned
//! exhaustively: no COMDATs, nine live reloc types, INIT_FUNCS in two libc
//! members. Out of scope (v1): gc-sections, debug sections (dropped on load),
//! comdats, PIC (`R_WASM_MEMORY_ADDR_REL_SLEB` in an actually-selected member is
//! a hard error naming the member).
//!
//! Differential contract: for every program in the confidence C5 corpus,
//! riwl-linked output must behave identically to rust-lld-linked output.

pub mod archive;
pub mod cli;
pub mod emit;
pub mod link;
pub mod object;

pub use link::{link, LinkOptions};

// --- byte-stream helpers shared by emit + reloc patching ---

pub(crate) fn uleb(out: &mut Vec<u8>, mut v: u64) {
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

pub(crate) fn sleb(out: &mut Vec<u8>, mut v: i64) {
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

/// Patch a 5-byte padded ULEB in place (reloc sites are always padded).
pub(crate) fn patch_uleb5(buf: &mut [u8], off: usize, v: u32) {
    buf[off] = (v & 0x7f) as u8 | 0x80;
    buf[off + 1] = ((v >> 7) & 0x7f) as u8 | 0x80;
    buf[off + 2] = ((v >> 14) & 0x7f) as u8 | 0x80;
    buf[off + 3] = ((v >> 21) & 0x7f) as u8 | 0x80;
    buf[off + 4] = ((v >> 28) & 0x0f) as u8;
}

/// Patch a 5-byte padded SLEB in place (i32 value range).
pub(crate) fn patch_sleb5(buf: &mut [u8], off: usize, v: i32) {
    buf[off] = (v & 0x7f) as u8 | 0x80;
    buf[off + 1] = ((v >> 7) & 0x7f) as u8 | 0x80;
    buf[off + 2] = ((v >> 14) & 0x7f) as u8 | 0x80;
    buf[off + 3] = ((v >> 21) & 0x7f) as u8 | 0x80;
    buf[off + 4] = ((v >> 28) & 0x7f) as u8;
}

pub(crate) fn patch_le32(buf: &mut [u8], off: usize, v: u32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

// --- relocation type constants (tool-conventions Linking.md) ---
pub(crate) const R_WASM_FUNCTION_INDEX_LEB: u8 = 0;
pub(crate) const R_WASM_TABLE_INDEX_SLEB: u8 = 1;
pub(crate) const R_WASM_TABLE_INDEX_I32: u8 = 2;
pub(crate) const R_WASM_MEMORY_ADDR_LEB: u8 = 3;
pub(crate) const R_WASM_MEMORY_ADDR_SLEB: u8 = 4;
pub(crate) const R_WASM_MEMORY_ADDR_I32: u8 = 5;
pub(crate) const R_WASM_TYPE_INDEX_LEB: u8 = 6;
pub(crate) const R_WASM_GLOBAL_INDEX_LEB: u8 = 7;
pub(crate) const R_WASM_MEMORY_ADDR_REL_SLEB: u8 = 11;
pub(crate) const R_WASM_TABLE_NUMBER_LEB: u8 = 20;

// --- symbol flags ---
pub(crate) const WASM_SYM_BINDING_WEAK: u32 = 0x01;
pub(crate) const WASM_SYM_BINDING_LOCAL: u32 = 0x02;
pub(crate) const WASM_SYM_UNDEFINED: u32 = 0x10;
