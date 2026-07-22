# clif2wasm — prototype: CLIF → waffle → wasm

*Prototype of `docs/plans/cranelift-wasm-backend.md` (Milestone A gates 1–2, plus a taste of
gate 3). Standalone: no cg_clif, no rustc.wasm, no browser — a plain Rust library tested
natively against wasmtime.*

## The question

Can CLIF (Cranelift IR, built for a 32-bit-pointer target) be translated through
[waffle](https://github.com/bytecodealliance/waffle) 0.3.0's public builder API and backend
(`validate()` → Reducifier → Ramsey stackify → treeify → localify) into valid,
correctly-executing wasm?

This is the plan's #1 stated unknown ("waffle unproven for a non-wasm frontend", §6/§16):
waffle's only production frontend is its own wasm parser (weval is wasm→wasm). If externally
built IR trips undocumented `validate()` invariants or backend assumptions, better to find
out in a week-scale prototype than in month two of the real crate.

## How it works

- `src/bridge.rs` — CLIF `Function` → waffle `FunctionBody`. Two passes, no placeholders:
  pass 1 creates blocks + blockparams (CLIF blockparams map 1:1 — both IRs are
  blockparam-SSA by design); pass 2 walks blocks in cranelift's own reverse post-order, so
  on reducible CFGs every def is translated before any use and a plain value map suffices.
  The CLIF entry block is mapped **onto** the entry block `FunctionBody::new` pre-creates
  (params bound, not re-added).
- `src/ops.rs` — the opcode lowering table. waffle's `Operator` is 1:1 with wasm opcodes,
  so this is instruction selection, not renaming: conditions normalize to i32 (eqz/eqz for
  i64), shift amounts get width fixups, i8/i16 fold to I32 (icmp results are i8 in
  cranelift 0.133 — the minimum unavoidable narrow-int handling).
- `src/module.rs` — module assembly: signatures (deduped; expanded so i128 → 2×i64) +
  `FuncDecl::None` placeholders first (so `Func` indices exist for direct calls), one
  linear memory, the `__stack_pointer` mutable global, one funcref table (slot 0 null,
  function k at k+1), translate bodies, export the entry, `Module::to_wasm_bytes()`.
- `src/i128_lower.rs` — i128 legalization to (lo, hi) i64 pairs: add/sub with
  carry/borrow synthesis, pairwise logic, hi/lo-combined comparisons, branchless
  dynamic shifts via `select`. The value map holds `One(v)` or `Pair(lo, hi)`; pairs
  flatten at every ABI position (blockparams, branch args, returns, call boundaries).
- **Shadow stack** (in `bridge.rs`): functions with sized stack slots get a prologue
  `sp_new = __stack_pointer - frame` + per-return restore; `stack_addr`/`stack_load`/
  `stack_store` lower against `sp_new`. Plain global ops — no waffle support needed.
- `tests/captured/` — **real cg_clif dumps** (see its README for the capture recipe),
  parsed with `cranelift-reader` and run through the same translate+execute harness.
- **Pointer-width stance:** no ISA, no `TargetFrontendConfig`. The wasm32 (U32) decision is
  expressed as input validation: loads/stores **reject non-i32 addresses**. CLIF captured
  from a 64-bit target must fail loudly, never truncate — a 32-bit frontend over
  64-bit-target CLIF silently miscompiles (plan §8). A negative test asserts the guard.

Calls resolve via the prototype convention `ExternalName::User { namespace: 0, index: k }`
where `k` is the callee's ordinal in the slice passed to `translate_module`.

## Reproduce

```
cargo test
```

Test inputs are hand-built with `cranelift-frontend` (the `spikes/c1-cranelift-on-wasm`
pattern, minus the ISA/`ctx.compile()` half — which is exactly what this crate replaces).
Every case round-trips `wasmparser` validation **and** executes under wasmtime.

## Results — ANSWER: YES (2026-07-20, two rounds)

All 21 tests pass: emitted modules are `wasmparser`-clean and execute correctly under
wasmtime 47. **waffle's `validate()` and backend accepted externally-built IR on the first
complete attempt in both rounds** — every compile failure during bring-up was
cranelift-side API churn, none were waffle invariants. The plan doc's "waffle unproven
for a non-wasm frontend" unknown is now *known-with-recipe*, including i128 pairs,
shadow-stack frames, indirect calls, and **real cg_clif-emitted CLIF as input**.

| Tier | What it proves | Test | Status |
|---|---|---|---|
| T1 | leaf arithmetic (i64/i32 ops, constants) | `t1_add_i64`, `t1_chain_i32` | PASS |
| T2 | icmp + select, straight-line | `t2_max_i64` | PASS |
| T3 | diamond + join blockparam | `t3_abs_i32` | PASS |
| T4 | **loop backedge through Ramsey (the crux)** — `sum(100)==5050` | `t4_sum_loop_i64` | PASS |
| T5 | *nested* structure: diamond inside loop | `t5_sum_even_i32` | PASS |
| T6 | memory + offset; pointer-width guard fires on i64 addr | `t6_*` | PASS |
| T7 | two functions, direct call, `Func` index resolution | `t7_call_double_loop` | PASS |
| T8 | f64 arithmetic (stretch) | `t8_hypot2_f64` | PASS |
| T9 | **i128 as (lo,hi) i64 pairs**: add/sub carry, all-cc compares, dynamic shl/ushr/sshr, iconcat/isplit | `t9_*` | PASS |
| T10 | **shadow stack**: `__stack_pointer` global, frame per call, slot isolation under recursion | `t10_fact_stack` | PASS |
| T11 | **function pointers**: `func_addr` → table index, `call_indirect` through the funcref table | `t11_call_indirect` | PASS |
| T12 | i128 across a call boundary (expanded multivalue ABI + `PickOutput` exercised for real) | `t12_call_i128` | PASS |
| T13 | **real cg_clif output** (tests/captured/): collatz, popcount, fib translate AND run; gcd's panic path fails loudly as the documented gap | `t13_*` | PASS |

Round 3 (see `docs/spikes/round3-*.md`; wasm32 captures via `spikes/s1-wasm32-capture`):

| Tier | What it proves | Test | Status |
|---|---|---|---|
| T14 | **data segments**: statics bump-allocated from 0x1000, `symbol_value` lowering, layout respects alignment | `t14_*` | PASS |
| T15 | **sret** (real captured `make_point`) + struct offsets (`point_sum` reading a static Point) | `t15_*` | PASS |
| T16 | **narrow-int discipline** on real cg_clif output: packed `[u8;3]` ABI, `load.i8`, u8 wrapping | `t16_*` | PASS |
| T17 | **whole CGU**: 7 real functions at their real FuncIds; `quad`→`double` calls; bounds-check + rem-by-zero panic paths trap via stubs; zero skipped dumps | `t17_*` | PASS |
| seam | **`cranelift_module::Module` impl** driven exactly as cg_clif drives it; Pulley32 `isa()` with `pointer_type()==I32`; `finish()` → executing wasm | `seam_*` (`--features module-seam`) | PASS |

## Friction / gap log

This list is a deliverable: it seeds the real crate's `ops.rs` and the plan's risk
register.

- **waffle friction: none at this scope.** No `validate()` rejections, no backend
  panics, no undocumented invariants hit. The entry-block param binding and
  `set_terminator`-once discipline (below) were derived by reading waffle's source
  up front — doing that appears to be sufficient.
- **cranelift 0.133 API churn hit the *test builders*, not the bridge**: `MemFlags`
  became an interned `u16` handle into a per-function `MemFlagsSet` (`ins().load` now
  takes `Into<MemFlagsData>`), and `ExtFuncData` grew a `patchable` field. Exactly the
  pre-1.0 churn the plan's version-discipline section predicts.
- Shift-amount width fixup: wasm requires amount type == operand type; CLIF doesn't.
  `I32WrapI64` / `I64ExtendI32U` inserted (both sides mask mod bit-width, semantics agree).
- Condition normalization: CLIF `brif`/`select` accept any int width; wasm `br_if`/`select`
  want i32. i8 (icmp results) folds via the type map; i64 conditions get `eqz`+`eqz`.
- `waffle 0.3.0` has no `waffle::compile(&Module)` free function (plan doc's phrasing is
  stale) — the entry point is `Module::to_wasm_bytes()`.
- `FunctionBody::new` pre-creates the entry block with signature-matching blockparams; the
  bridge must bind the CLIF entry params onto them rather than adding its own.
- `set_terminator` maintains succ/pred edge lists itself (and asserts one-terminator-per-
  block) — no `recompute_edges()` needed for straight-line building.
- Dep pins: `wasmtime = "47"` deliberately — wasmtime 46 requires `cranelift ^0.133.x`,
  which cargo cannot unify with our `=0.133.0` pin; 47 uses `^0.134.x` (a different 0.x
  line, so two independent cranelift copies coexist).
- **cg_clif capture recipe**: `rustup component add rustc-codegen-cranelift-preview`,
  then `rustc -Zcodegen-backend=cranelift --emit llvm-ir` dumps per-function CLIF into
  `<crate>.clif/<fn>.{unopt,opt}.clif` (the trailing "could not copy …rcgu.ll" error is
  cosmetic). The `.unopt.clif` is what `define_function` receives.
- **Version skew is live**: the nightly's bundled cg_clif uses a newer cranelift than
  0.133. Concretely, `icmp_imm` is builder-only sugar in 0.133 (expands to
  `iconst`+`icmp`, see 0.133 `ir/builder.rs`) but a real printed instruction upstream —
  so `cranelift-reader` 0.133 can't parse newer dumps verbatim. The test helper performs
  0.133's own expansion textually. This is the plan's pin-discipline warning made flesh:
  the real crate must pin cranelift to the *exact* cg_clif it forks.
- **What real cg_clif emits** that hand-built IR didn't: `nop` anchors (skipped), long
  alias chains (`resolve_aliases` handles), raw-integer `brif` conditions, `icmp_imm`,
  `udiv`/`urem`. Next genuine gaps surfaced by the gcd panic path: `symbol_value` of
  data symbols (needs data segments / gate A.4) and calls into `core::panicking` (needs
  multi-CGU name resolution beyond the ordinal convention).
- Shadow stack needed no waffle support at all: a mutable I32 global + prologue
  decrement + per-return restore, emitted as ordinary ops in the entry/return blocks.
- i128 pair lowering (incl. multivalue returns and `PickOutput` projections at call
  boundaries) passed waffle validation first try; `imul.i128` is deliberately on the
  gap list.
- **Round 3 additions**: cg_clif rejects wasm32 targets outright (`build_isa` — see
  `docs/spikes/round3-s1-wasm32-capture-DONE.md`); the pulley32 capture rig yields real
  i32-pointer CLIF; narrow ints (i8/i16) forced into scope by real dumps → canonical
  zero-extended representation; **reader gotcha**: `userextnameN` gv refs in parsed
  dumps can falsely resolve to callee entries (reader rebuilds `user_named_funcs` from
  `u0:M` tokens) — the comment-borne side table must take precedence; `ModuleDeclarations`
  provides all declare/dedup/linkage bookkeeping for free; a Pulley32 `TargetIsa`
  satisfies cg_clif's pointer-width invariant for wasm32 with zero core changes.
- **Fork-scope gaps enumerated** (the real crate's work list): data relocs (function/
  data pointers inside statics), data imports, `define_function_bytes`, TLS,
  `imul.i128`, `br_table` (never forced by any real dump so far), non-`Symbol`
  global-value chains, real cross-CGU imports (linker scope).

## Non-goals (scope fence)

Round two pulled i128 (minus `imul.i128`), the shadow stack, `call_indirect` +
funcref tables, and multivalue/i128 call ABI **into** scope. Still fenced out:

`cranelift_module::Module` impl; cg_clif *integration* (we consume its dumps, we don't
fork its driver); rustc.wasm/toolchain work; `imul.i128`; sret; data segments / statics
(`symbol_value` — the gcd gap); relocatable objects / linker; wasip1 / std; browser /
runner.js; irreducible CFGs; trapping fcvt; NaN fine points; narrow-int legalization
beyond the i8-icmp-condition minimum; `br_table`; exporting memory; cross-CGU
`ExternalName` resolution (panic calls into core).

Anything not in the starter operator set fails with a clear `unimplemented opcode` error —
the gap list those errors produce is a deliverable, not a failure.
