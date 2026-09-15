# Proving under a memory ceiling

Why `midnight-proofs` can map its SRS and prover key, and spill its cosets, and
what that costs.

This is written for someone deciding whether to accept these changes, so it
leads with the measurements — including the ones that are unflattering — and it
is explicit about what has not been measured.

## The problem

A PLONK prover's peak memory is dominated by things that are not the proof:
the structured reference string, the prover key, and the extended-domain cosets
built during `compute_h_poly`. At k=20 on BLS12-381 those add up to roughly
14 GiB of anonymous heap.

On a server that is unremarkable. On a phone it is fatal, and not because the
device has no RAM — it is because **anonymous pages cannot be reclaimed**. Both
Android's low-memory killer and iOS's jetsam act on a process's private dirty
footprint. A prover holding 14 GiB of heap is killed; a prover holding the same
data in file-backed pages is not, because the kernel can evict and re-fault
them under pressure.

So the goal is not to use less memory. It is to move memory the prover must
have into a form the operating system is allowed to take back.

## What was measured

M-series laptop, release build, the ZSwap output circuit, one proof at a time.
"First" means the first proof *after* SRS and prover key are ready — not
process start.

| k | policy | first | warm | memory | spill disk |
|---:|---|---:|---:|---|---:|
| 14 | heap | — | 0.771 s | not sampled | 0 |
| 14 | mapped key | — | 0.784 s | not sampled | key sidecar |
| 14 | mapped key + spilled cosets | — | 1.519 s | not sampled | not sampled |
| 19 | heap | 23.77 s | 25.21 s | 7.26 GiB RSS | 0 |
| 19 | mapped key + spilled cosets | 38.34 s | 37.01 s | 6.70 GiB RSS | 4.96 GiB |
| 20 | heap | 45.5–51.1 s | 50.20 s | 14.26 GiB RSS / 14.26 GiB footprint | 0 |
| 20 | mapped key + spilled cosets | 71.53 s | 66.19 s | 13.54 GiB RSS / **6.04 GiB footprint** | 9.89 GiB |

Read the k=20 row carefully, because it is the whole argument. **RSS barely
moves** — 14.26 to 13.54 GiB — since RSS counts file-backed pages. The number
the platform acts on, macOS physical footprint, falls by about **58%**. Quoting
RSS here would hide the entire benefit; quoting footprint on a platform that
has no such metric would be meaningless.

And the cost is real: **about 47% slower at k=19, 32% at k=20.** This is not a
speed optimisation. It buys reclaimable memory with CPU and disk I/O, and on a
host where RAM is not the binding constraint it is a straight loss. At k=14 the
mapped key is free (within the interval) and forcing the coset spill roughly
doubles the time — which is why the spill has a floor and does not engage below
k=18 by default.

### What has not been measured

- **Anything on a real device.** The laptop has no memory pressure and no
  jetsam. A prior physical-device run completed k=20 at 4,393 MiB HWM where the
  heap path reached ~6.8 GiB and failed, but it used an earlier, unsound
  implementation and has not been repeated.
- **Anything under concurrency**, or under a cgroup limit.
- **Cold start.** The table begins after keys are ready.

## The approach

Three independent capabilities, each off by default.

**Mapped SRS.** `ParamsKZG::read_mmap_arc` builds parameters whose `g` and
`g_lagrange` are slice views into a mapped companion file rather than heap
`Vec`s. Backed by `BasesStorage<C>`, an enum of `Owned(Vec<C>)` or
`Mapped { mmap, ptr, len }` with a `Deref` to `[C]`, so consumers are unchanged.

**Mapped prover key.** The same idea for `fixed_polys`, `fixed_values` and
`permutation.polys`.

**Spilled cosets.** `compute_h_poly` streams each extended-domain coset to a
temporary file and drops it before building the next, then maps the file back.
Peak transient heap becomes roughly one coset rather than all of them.

All three are behind `mmap` and `disk-spill` features, off by default, and the
policy is a value — `config::ProverConfig` — not a set of environment reads.
A `config::ProverContext` carries that value plus an optional `CancelToken`
through `ProvingKey::read_with_policy` and `plonk::create_proof_with`, so two
proofs in one process can hold different policies and be cancelled
independently; cancellation is `Error::Cancelled`, not a panic. The
context-free entry points still exist and run under the environment's policy.

## Alternatives considered, and why not

**Just use less memory.** The SRS, the prover key and the cosets are all
required by the algorithm. There is no version of this that needs fewer bytes.

**Recompute instead of storing.** `read_custom_lazy` does exactly this for
`g_lagrange`, trading an inverse NTT for the storage. It is included, and it is
the right trade only when the recompute amortises over a long idle period. In a
tight prove loop it is a large loss.

**A borrowed `Polynomial`.** The spill originally built `Polynomial` values
whose `values: Vec<F>` pointed into the mapping, via `Vec::from_raw_parts`.
**That is undefined behaviour** — the contract requires a pointer from the
global allocator — and wrapping the result in `ManuallyDrop` addresses the wrong
half of it. Changing `Polynomial` to carry an enum storage would fix it but
adds an `F: 'static` bound to a public type and touches ~49 call sites.

The shipped answer is narrower: a crate-private `PolynomialRead` /
`PolynomialView` seam. Readers take a view; mapped data is `Arc<Mmap>` plus a
checked offset and length; the public `Polynomial` is untouched. Offsets rather
than raw pointers, so moving or cloning a descriptor cannot invalidate it, and
`Send`/`Sync` derive rather than needing `unsafe impl`.

**One umbrella feature.** Rejected because the capabilities genuinely differ:
WASI has a filesystem but no `mmap`; browser wasm has neither. An umbrella
would force a target to take both or neither. `disk-spill` enables `mmap`
because spilling has to map back what it wrote.

## What a reviewer should check

- **Both arms agree.** An optimisation that changes a proof is worse than no
  optimisation. `plonk::cosets::spilled_and_heap_cosets_agree` and
  `poly::kzg::msm::chunked_msm_agrees_with_unchunked` exist for this, and test
  counts rise with features — 43 / 54 / 65 — so a gated-off arm cannot go
  quietly untested. `dev::cost_model`'s real-key test proves with a *spilled*
  key and verifies with the verifier that accepts the heap key's proof.
- **The `unsafe` is confined and opted out of per site**, not per module, so a
  newly added `unsafe` block is a build error rather than a silent addition.
- **Failure is not silent.** A spill that cannot be created falls back to the
  heap and logs; under `bench-internal` it panics, so a benchmark cannot
  measure the heap arm and call it a spill. Backing store is preallocated
  (`posix_fallocate`, `F_PREALLOCATE`) so ENOSPC surfaces as an `io::Error`
  rather than a SIGBUS during a page fault.
- **The companion file is a local cache, not an interchange format.** Its
  header is validated — magic, version, a layout identity over (curve type,
  point size, alignment, `k`), offsets, counts, alignment — with every length
  and offset through checked arithmetic, so a damaged file is `InvalidData`,
  never a panic and never an out-of-range slice. Its *contents* are not
  validated: the identity tells a file built for a different curve, layout or
  `k` from this one, not a same-`k` SRS from a different setup, so a file an
  attacker can write yields wrong proofs. Both entry points are bounded on the
  sealed `PlainBytes` trait, so the byte reinterpretation is only reachable for
  curves whose layout this crate has checked. The trust boundary is stated on
  both entry points. The layout the `PlainBytes` argument quotes is also
  pinned by compile-time assertions next to the `impl`, so a `blst`, compiler
  or target change that alters it fails the build there rather than being
  caught only by the identity at the first read.
- **A misconfigured spill directory fails loudly and names itself.** A key
  load whose `spill_dir` cannot take a temp file is rejected whole — no
  half-spilled key, no silent fallback to the heap — and the error carries the
  operation and the directory (`create spill temp file in /spill: …`), so an
  operator can tell a bad volume from a bad request. Two deployment facts
  learned the hard way: on Docker Desktop a **bind-mounted host directory**
  makes that temp-file creation fail with `ENOENT` even though the directory
  exists (use a named volume or a VM-local path), and `tmpfs` works but is
  memory-backed, which defeats what spill is for.

## What is deliberately not here

Policy beyond the crate: choosing a profile per device class, admission
control, and telemetry belong to the consumer. `ProverContext` is the seam; the
decision is not ours to make from inside a proving library.

One knob is deliberately still process-wide: the MSM chunk size. It sits under
`PolynomialCommitmentScheme::commit(params, poly)`, a public trait with no
per-call seam, and widening that trait for a tuning knob would be a bigger
change than the knob is worth. It is validated before the shift instead, and
documented as the exception.
