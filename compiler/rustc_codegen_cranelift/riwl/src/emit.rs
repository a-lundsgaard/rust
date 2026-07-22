//! Final module emission (hand-rolled bytes, like clif2wasm's object writer).

use crate::uleb;

pub struct Emitter {
    pub out: Vec<u8>,
}

impl Emitter {
    pub fn new() -> Emitter {
        Emitter { out: b"\0asm\x01\0\0\0".to_vec() }
    }

    pub fn section(&mut self, id: u8, payload: &[u8]) {
        self.out.push(id);
        uleb(&mut self.out, payload.len() as u64);
        self.out.extend_from_slice(payload);
    }
}

pub fn write_name(out: &mut Vec<u8>, s: &str) {
    uleb(out, s.len() as u64);
    out.extend_from_slice(s.as_bytes());
}

pub fn valtype_byte(t: wasmparser::ValType) -> u8 {
    use wasmparser::ValType::*;
    match t {
        I32 => 0x7f,
        I64 => 0x7e,
        F32 => 0x7d,
        F64 => 0x7c,
        V128 => 0x7b,
        Ref(_) => 0x70, // only funcref appears in our universe
    }
}
