//! wasm-ld-compatible argument parsing (the flag subset rustc's wasip1 link
//! line uses), shared by the CLI binary and the in-rustc library entry.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};

use crate::LinkOptions;

/// Parse wasm-ld-style args (NOT including argv[0]) and run the link.
pub fn run_args(argv: Vec<String>) -> Result<()> {
    let mut raw = argv;
    // Response files (@path) — rustc uses them when the line gets long.
    let mut args = Vec::new();
    while let Some(a) = raw.first().cloned() {
        raw.remove(0);
        if let Some(path) = a.strip_prefix('@') {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("response file {path}"))?;
            let mut expanded: Vec<String> = Vec::new();
            for tok in split_response(&content) {
                expanded.push(tok);
            }
            expanded.extend(raw);
            raw = expanded;
        } else {
            args.push(a);
        }
    }

    let mut opts = LinkOptions::default();
    let mut out: Option<PathBuf> = None;
    let args_for_diag = args.clone();
    let mut it = args.into_iter().peekable();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-flavor" => {
                let f = it.next().context("-flavor needs a value")?;
                if f != "wasm" {
                    bail!("unsupported flavor {f}");
                }
            }
            "-o" => out = Some(PathBuf::from(it.next().context("-o needs a value")?)),
            "--no-entry" => opts.no_entry = true,
            "--export" => opts.exports.push(it.next().context("--export needs a value")?),
            s if s.starts_with("--export=") => {
                opts.exports.push(s["--export=".len()..].to_string())
            }
            "-z" => {
                let v = it.next().context("-z needs a value")?;
                if let Some(n) = v.strip_prefix("stack-size=") {
                    opts.stack_size = n.parse().context("stack-size")?;
                }
                // other -z options ignored
            }
            s if s.starts_with("-zstack-size=") => {
                opts.stack_size = s["-zstack-size=".len()..].parse().context("stack-size")?;
            }
            "--stack-first" => opts.stack_first = true,
            "--allow-undefined" | "--import-undefined" => opts.allow_undefined = true,
            "-L" => opts
                .lib_paths
                .push(PathBuf::from(it.next().context("-L needs a value")?)),
            s if s.starts_with("-L") && s.len() > 2 => {
                opts.lib_paths.push(PathBuf::from(&s[2..]))
            }
            "-l" => opts.libs.push(it.next().context("-l needs a value")?),
            s if s.starts_with("-l") && s.len() > 2 => opts.libs.push(s[2..].to_string()),
            // Accepted and ignored (semantics-free for our output or defaults).
            "--no-demangle" | "--gc-sections" | "--strip-debug" | "-s" | "--strip-all"
            | "--eh-frame-hdr" | "--fatal-warnings" | "--as-needed" => {}
            s if s.starts_with("--export-memory") => {}
            s if s.starts_with("-O") => {}
            s if s.starts_with('-') => {
                bail!("unsupported flag {s} (full argv: {:?})", args_for_diag)
            }
            path => opts.inputs.push(PathBuf::from(path)),
        }
    }

    let out = out.context("missing -o")?;
    let bytes = crate::link(&opts)?;
    std::fs::write(&out, &bytes).with_context(|| format!("writing {}", out.display()))?;
    Ok(())
}

/// Split a response file: whitespace-separated, with quoting per lld rules
/// (double/single quotes, backslash escapes).
fn split_response(s: &str) -> Vec<String> {
    let mut toks = Vec::new();
    let mut cur = String::new();
    let mut chars = s.chars().peekable();
    let mut in_tok = false;
    while let Some(c) = chars.next() {
        match c {
            c if c.is_whitespace() => {
                if in_tok {
                    toks.push(std::mem::take(&mut cur));
                    in_tok = false;
                }
            }
            '"' | '\'' => {
                in_tok = true;
                let q = c;
                for c2 in chars.by_ref() {
                    if c2 == q {
                        break;
                    }
                    cur.push(c2);
                }
            }
            '\\' => {
                in_tok = true;
                if let Some(c2) = chars.next() {
                    cur.push(c2);
                }
            }
            c => {
                in_tok = true;
                cur.push(c);
            }
        }
    }
    if in_tok {
        toks.push(cur);
    }
    toks
}
