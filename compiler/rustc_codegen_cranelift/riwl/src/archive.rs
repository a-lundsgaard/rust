//! Minimal `ar` archive reader (GNU variant — what rlibs and wasi-libc use).
//! Members that aren't wasm objects (lib.rmeta, symbol tables) are skipped.

use anyhow::{bail, ensure, Context, Result};

pub struct ArchiveMember {
    pub name: String,
    /// Byte range of the member's contents within the archive file.
    pub range: (usize, usize),
}

pub fn parse_archive(name: &str, bytes: &[u8]) -> Result<Vec<ArchiveMember>> {
    ensure!(bytes.starts_with(b"!<arch>\n"), "{name}: not an ar archive");
    let mut off = 8usize;
    let mut extended_names: &[u8] = &[];
    let mut members = Vec::new();
    while off + 60 <= bytes.len() {
        let hdr = &bytes[off..off + 60];
        ensure!(&hdr[58..60] == b"`\n", "{name}: bad ar header at {off}");
        let raw_name = std::str::from_utf8(&hdr[0..16]).context("member name")?.trim_end();
        let size: usize = std::str::from_utf8(&hdr[48..58])
            .context("size")?
            .trim_end()
            .parse()
            .with_context(|| format!("{name}: member size at {off}"))?;
        let data_start = off + 60;
        ensure!(data_start + size <= bytes.len(), "{name}: truncated member");

        if raw_name == "/" || raw_name == "<SYMDEF>" || raw_name.starts_with("__.SYMDEF") {
            // symbol index — we build our own
        } else if raw_name == "//" {
            extended_names = &bytes[data_start..data_start + size];
        } else {
            let member_name = if let Some(idx) = raw_name.strip_prefix('/') {
                // GNU long-name reference: offset into the // member, NUL/newline
                // terminated entries of the form "name/\n".
                let start: usize = idx.parse().with_context(|| format!("{name}: longname"))?;
                let rest = extended_names.get(start..).context("longname offset")?;
                let end = rest
                    .iter()
                    .position(|&b| b == b'\n' || b == 0)
                    .unwrap_or(rest.len());
                let s = std::str::from_utf8(&rest[..end]).context("longname utf8")?;
                s.trim_end_matches('/').to_string()
            } else {
                raw_name.trim_end_matches('/').to_string()
            };
            members.push(ArchiveMember { name: member_name, range: (data_start, data_start + size) });
        }

        off = data_start + size;
        if off % 2 == 1 {
            off += 1; // members are 2-byte aligned
        }
    }
    if members.is_empty() {
        bail!("{name}: no members");
    }
    Ok(members)
}

pub fn is_wasm(bytes: &[u8]) -> bool {
    bytes.starts_with(b"\0asm")
}
