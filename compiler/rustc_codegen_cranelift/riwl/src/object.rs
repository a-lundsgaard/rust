//! Parse one relocatable wasm object into the linker's model.

use anyhow::{bail, ensure, Context, Result};
use wasmparser::{BinaryReader, Linking, LinkingSectionReader, Parser, Payload, TypeRef};

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct FuncType {
    pub params: Vec<wasmparser::ValType>,
    pub results: Vec<wasmparser::ValType>,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum SymKind {
    Func,
    Data,
    Global,
    Table,
}

#[derive(Clone, Debug)]
pub enum SymDef {
    /// wasm function index in THIS object (imports first).
    Func { local_index: u32 },
    Data { segment: u32, offset: u32, size: u32 },
    Global { local_index: u32 },
    Table { local_index: u32 },
}

#[derive(Clone, Debug)]
pub struct Symbol {
    pub name: String,
    pub kind: SymKind,
    pub flags: u32,
    pub def: Option<SymDef>,
    /// For undefined func/global/table symbols: the (module, field) of the
    /// corresponding import entry.
    pub import: Option<(String, String)>,
    /// For undefined func symbols: the import's type. Used to validate
    /// resolution (signature mismatch = the ABI tripwire).
    pub func_type: Option<FuncType>,
}

impl Symbol {
    pub fn is_undefined(&self) -> bool {
        self.def.is_none()
    }
    pub fn is_weak(&self) -> bool {
        self.flags & crate::WASM_SYM_BINDING_WEAK != 0
    }
    pub fn is_local(&self) -> bool {
        self.flags & crate::WASM_SYM_BINDING_LOCAL != 0
    }
}

#[derive(Clone, Debug)]
pub struct Segment {
    pub name: String,
    pub align_log2: u32,
    /// Range of the segment's data bytes inside `data_payload`.
    pub payload_off: u32,
    pub len: u32,
}

#[derive(Copy, Clone, Debug)]
pub struct RelocEntry {
    pub ty: u8,
    pub offset: u32,
    pub index: u32,
    pub addend: i64,
}

pub struct InputObject {
    pub name: String,
    pub types: Vec<FuncType>,
    /// (module, field, type) of each FUNCTION import, in order.
    pub func_imports: Vec<(String, String, u32)>,
    /// (module, field) of each GLOBAL / TABLE import, in order.
    pub global_imports: Vec<(String, String)>,
    pub table_imports: Vec<(String, String)>,
    /// Defined functions' local type indices, in order.
    pub func_type_idx: Vec<u32>,
    /// The raw code-section payload (count + size-prefixed entries); relocs are
    /// applied IN PLACE here, then bodies re-sliced for emission.
    pub code_payload: Vec<u8>,
    /// (start, len) of each body's bytes (locals+code, no size prefix) within
    /// `code_payload`, in definition order.
    pub code_body_ranges: Vec<(u32, u32)>,
    /// The raw data-section payload; relocs applied in place.
    pub data_payload: Vec<u8>,
    pub segments: Vec<Segment>,
    pub symbols: Vec<Symbol>,
    pub code_relocs: Vec<RelocEntry>,
    pub data_relocs: Vec<RelocEntry>,
    /// (priority, symbol index) from WASM_INIT_FUNCS.
    pub init_funcs: Vec<(u32, u32)>,
}

impl InputObject {
    pub fn func_import_count(&self) -> u32 {
        self.func_imports.len() as u32
    }

    /// Parse `bytes` as a relocatable object. `name` is for diagnostics.
    pub fn parse(name: &str, bytes: &[u8]) -> Result<InputObject> {
        let mut types = Vec::new();
        let mut func_imports = Vec::new();
        let mut global_imports = Vec::new();
        let mut table_imports = Vec::new();
        let mut func_type_idx = Vec::new();
        let mut code_payload = Vec::new();
        let mut code_body_ranges = Vec::new();
        let mut data_payload = Vec::new();
        let mut seg_ranges: Vec<(u32, u32)> = Vec::new(); // (payload_off, len)
        let mut symbols_raw: Vec<RawSym> = Vec::new();
        let mut seg_infos: Vec<(String, u32)> = Vec::new();
        let mut code_relocs = Vec::new();
        let mut data_relocs = Vec::new();
        let mut init_funcs = Vec::new();
        // Section ordinal -> is it code / data (for reloc target resolution).
        let mut section_kinds: Vec<u8> = Vec::new(); // wasm section id per ordinal
        let mut code_section_start = 0usize;

        struct RawSym {
            kind: SymKind,
            flags: u32,
            index: u32, // func/global/table index, or segment for defined data
            offset: u32,
            size: u32,
            name: Option<String>,
        }

        for payload in Parser::new(0).parse_all(bytes) {
            let payload = payload.with_context(|| format!("parsing {name}"))?;
            match &payload {
                Payload::TypeSection(r) => {
                    section_kinds.push(1);
                    for group in r.clone() {
                        for t in group?.into_types() {
                            let wasmparser::CompositeInnerType::Func(f) = &t.composite_type.inner
                            else {
                                bail!("{name}: non-func type");
                            };
                            types.push(FuncType {
                                params: f.params().to_vec(),
                                results: f.results().to_vec(),
                            });
                        }
                    }
                }
                Payload::ImportSection(r) => {
                    section_kinds.push(2);
                    for group in r.clone() {
                        for item in group? {
                            let (_off, imp) = item?;
                            match imp.ty {
                                TypeRef::Func(t) | TypeRef::FuncExact(t) => func_imports.push((
                                    imp.module.to_string(),
                                    imp.name.to_string(),
                                    t,
                                )),
                                TypeRef::Global(_) => global_imports
                                    .push((imp.module.to_string(), imp.name.to_string())),
                                TypeRef::Table(_) => table_imports
                                    .push((imp.module.to_string(), imp.name.to_string())),
                                TypeRef::Memory(_) => {} // single linear memory, index 0
                                TypeRef::Tag(_) => bail!("{name}: tag import"),
                            }
                        }
                    }
                }
                Payload::FunctionSection(r) => {
                    section_kinds.push(3);
                    for t in r.clone() {
                        func_type_idx.push(t?);
                    }
                }
                Payload::TableSection(_) => {
                    section_kinds.push(4);
                    bail!("{name}: defined table in relocatable object");
                }
                Payload::MemorySection(_) => {
                    section_kinds.push(5);
                }
                Payload::GlobalSection(_) => {
                    section_kinds.push(6);
                    bail!("{name}: defined global in relocatable object");
                }
                Payload::ExportSection(_) => {
                    section_kinds.push(7);
                }
                Payload::StartSection { .. } => {
                    section_kinds.push(8);
                }
                Payload::ElementSection(r) => {
                    section_kinds.push(9);
                    // Input element segments only DECLARE address-taken functions
                    // (either declarative segments, or LLVM's pre-reference-types
                    // active segment at offset 1). The authoritative table
                    // information is the R_WASM_TABLE_INDEX_* relocations — the
                    // output table is rebuilt from those, so input contents are
                    // ignored either way.
                    for seg in r.clone() {
                        let seg = seg?;
                        ensure!(
                            matches!(
                                seg.kind,
                                wasmparser::ElementKind::Declared
                                    | wasmparser::ElementKind::Active { .. }
                            ),
                            "{name}: passive element segment in relocatable object"
                        );
                    }
                }
                Payload::DataCountSection { .. } => {
                    section_kinds.push(12);
                }
                Payload::CodeSectionStart { range, .. } => {
                    section_kinds.push(10);
                    // Payload = everything after the section header (count + entries);
                    // reloc offsets are relative to this. Bodies re-sliced via
                    // code_body_ranges after in-place patching.
                    code_payload = bytes[range.start..range.end].to_vec();
                    code_body_ranges.clear();
                    code_section_start = range.start;
                }
                Payload::CodeSectionEntry(body) => {
                    let r = body.range();
                    code_body_ranges.push((
                        (r.start - code_section_start) as u32,
                        (r.end - r.start) as u32,
                    ));
                }
                Payload::DataSection(r) => {
                    section_kinds.push(11);
                    let range = r.range();
                    data_payload = bytes[range.start..range.end].to_vec();
                    for seg in r.clone() {
                        let seg = seg?;
                        let wasmparser::DataKind::Active { memory_index: 0, offset_expr: _ } =
                            seg.kind
                        else {
                            bail!("{name}: non-active data segment");
                        };
                        let start = seg.range.end - seg.data.len() - range.start;
                        seg_ranges.push((start as u32, seg.data.len() as u32));
                    }
                }
                Payload::CustomSection(c) => {
                    section_kinds.push(0);
                    if c.name() == "linking" {
                        let br = BinaryReader::new(c.data(), c.data_offset());
                        let lr = LinkingSectionReader::new(br)?;
                        ensure!(lr.version() == 2, "{name}: linking version {}", lr.version());
                        for sub in lr.subsections() {
                            match sub? {
                                Linking::SymbolTable(map) => {
                                    for s in map {
                                        use wasmparser::SymbolInfo as SI;
                                        symbols_raw.push(match s? {
                                            SI::Func { flags, index, name } => RawSym {
                                                kind: SymKind::Func,
                                                flags: flags.bits(),
                                                index,
                                                offset: 0,
                                                size: 0,
                                                name: name.map(|s| s.to_string()),
                                            },
                                            SI::Data { flags, name, symbol } => RawSym {
                                                kind: SymKind::Data,
                                                flags: flags.bits(),
                                                index: symbol.map_or(0, |d| d.index),
                                                offset: symbol.map_or(0, |d| d.offset),
                                                size: symbol.map_or(0, |d| d.size),
                                                name: Some(name.to_string()),
                                            }
                                            .with_defined(symbol.is_some()),
                                            SI::Global { flags, index, name } => RawSym {
                                                kind: SymKind::Global,
                                                flags: flags.bits(),
                                                index,
                                                offset: 0,
                                                size: 0,
                                                name: name.map(|s| s.to_string()),
                                            },
                                            SI::Table { flags, index, name } => RawSym {
                                                kind: SymKind::Table,
                                                flags: flags.bits(),
                                                index,
                                                offset: 0,
                                                size: 0,
                                                name: name.map(|s| s.to_string()),
                                            },
                                            SI::Section { .. } => RawSym {
                                                kind: SymKind::Data,
                                                flags: crate::WASM_SYM_BINDING_LOCAL,
                                                index: u32::MAX,
                                                offset: 0,
                                                size: 0,
                                                name: Some(String::new()),
                                            },
                                            other => {
                                                bail!("{name}: unsupported symbol {other:?}")
                                            }
                                        });
                                    }
                                }
                                Linking::SegmentInfo(map) => {
                                    for seg in map {
                                        let seg = seg?;
                                        seg_infos
                                            .push((seg.name.to_string(), seg.alignment));
                                    }
                                }
                                Linking::InitFuncs(map) => {
                                    for f in map {
                                        let f = f?;
                                        init_funcs.push((f.priority, f.symbol_index));
                                    }
                                }
                                Linking::ComdatInfo(_) => {
                                    bail!("{name}: COMDATs are out of riwl v1 scope")
                                }
                                other => bail!("{name}: linking subsection {other:?}"),
                            }
                        }
                    } else if let Some(target) = c.name().strip_prefix("reloc.") {
                        let mut br = BinaryReader::new(c.data(), c.data_offset());
                        let _section = br.read_var_u32()?;
                        let count = br.read_var_u32()?;
                        let mut list = Vec::with_capacity(count as usize);
                        for _ in 0..count {
                            let ty = br.read_u8()?;
                            let offset = br.read_var_u32()?;
                            let index = br.read_var_u32()?;
                            let addend = match ty {
                                crate::R_WASM_MEMORY_ADDR_LEB
                                | crate::R_WASM_MEMORY_ADDR_SLEB
                                | crate::R_WASM_MEMORY_ADDR_I32
                                | crate::R_WASM_MEMORY_ADDR_REL_SLEB
                                | 8 /* FUNCTION_OFFSET_I32 */
                                | 9 /* SECTION_OFFSET_I32 */ => br.read_var_i64()? ,
                                _ => 0,
                            };
                            list.push(RelocEntry { ty, offset, index, addend });
                        }
                        match target {
                            "CODE" => code_relocs = list,
                            "DATA" => data_relocs = list,
                            // debug-section relocs: sections dropped, relocs too.
                            _ => {}
                        }
                    }
                    // all other custom sections (name/producers/target_features/
                    // .debug_*) are dropped.
                }
                _ => {}
            }
        }

        impl RawSym {
            fn with_defined(mut self, defined: bool) -> RawSym {
                if !defined {
                    self.flags |= crate::WASM_SYM_UNDEFINED;
                }
                self
            }
        }

        // Resolve raw symbols into the model.
        let n_func_imports = func_imports.len() as u32;
        let n_global_imports = global_imports.len() as u32;
        let n_table_imports = table_imports.len() as u32;
        let mut symbols = Vec::with_capacity(symbols_raw.len());
        for rs in symbols_raw {
            let undefined = rs.flags & crate::WASM_SYM_UNDEFINED != 0;
            let (def, import, func_type, name_str) = match rs.kind {
                SymKind::Func => {
                    if undefined {
                        ensure!(rs.index < n_func_imports, "{name}: undef func sym index");
                        let (m, f, t) = &func_imports[rs.index as usize];
                        let ft = types
                            .get(*t as usize)
                            .cloned()
                            .with_context(|| format!("{name}: import type"))?;
                        (
                            None,
                            Some((m.clone(), f.clone())),
                            Some(ft),
                            rs.name.unwrap_or_else(|| f.clone()),
                        )
                    } else {
                        let ti = if rs.index >= n_func_imports {
                            func_type_idx[(rs.index - n_func_imports) as usize]
                        } else {
                            func_imports[rs.index as usize].2
                        };
                        let ft = types.get(ti as usize).cloned();
                        (
                            Some(SymDef::Func { local_index: rs.index }),
                            None,
                            ft,
                            rs.name.context("defined func sym without name")?,
                        )
                    }
                }
                SymKind::Data => {
                    if rs.index == u32::MAX {
                        // section symbol — keep as an inert local
                        (None, None, None, rs.name.unwrap_or_default())
                    } else if undefined {
                        (None, None, None, rs.name.context("undef data sym name")?)
                    } else {
                        (
                            Some(SymDef::Data {
                                segment: rs.index,
                                offset: rs.offset,
                                size: rs.size,
                            }),
                            None,
                            None,
                            rs.name.context("data sym name")?,
                        )
                    }
                }
                SymKind::Global => {
                    if undefined {
                        ensure!(rs.index < n_global_imports, "{name}: undef global sym index");
                        let (m, f) = &global_imports[rs.index as usize];
                        (
                            None,
                            Some((m.clone(), f.clone())),
                            None,
                            rs.name.unwrap_or_else(|| f.clone()),
                        )
                    } else {
                        (
                            Some(SymDef::Global { local_index: rs.index }),
                            None,
                            None,
                            rs.name.context("global sym name")?,
                        )
                    }
                }
                SymKind::Table => {
                    if undefined {
                        ensure!(rs.index < n_table_imports, "{name}: undef table sym index");
                        let (m, f) = &table_imports[rs.index as usize];
                        (
                            None,
                            Some((m.clone(), f.clone())),
                            None,
                            rs.name.unwrap_or_else(|| f.clone()),
                        )
                    } else {
                        (
                            Some(SymDef::Table { local_index: rs.index }),
                            None,
                            None,
                            rs.name.context("table sym name")?,
                        )
                    }
                }
            };
            symbols.push(Symbol {
                name: name_str,
                kind: rs.kind,
                flags: rs.flags,
                def,
                import,
                func_type,
            });
        }

        ensure!(
            seg_infos.len() == seg_ranges.len() || seg_infos.is_empty(),
            "{name}: SEGMENT_INFO count {} != data segments {}",
            seg_infos.len(),
            seg_ranges.len()
        );
        let segments = seg_ranges
            .iter()
            .enumerate()
            .map(|(i, &(off, len))| {
                let (nm, al) = seg_infos
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| (format!(".seg{i}"), 0));
                Segment { name: nm, align_log2: al, payload_off: off, len }
            })
            .collect();

        Ok(InputObject {
            name: name.to_string(),
            types,
            func_imports,
            global_imports,
            table_imports,
            func_type_idx,
            code_payload,
            code_body_ranges,
            data_payload,
            segments,
            symbols,
            code_relocs,
            data_relocs,
            init_funcs,
        })
    }
}
