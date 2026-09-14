//! Coset construction for the prover, with an optional disk-spilled backing.
//!
//! Always compiled. The spilled arm exists only under the `disk-spill` feature,
//! so a wasm consumer gets the plain heap path with no filesystem dependency
//! reachable and none shipped — while the prover's call sites stay free of
//! `cfg`, which is where scattered gating would otherwise accumulate.

#[cfg(feature = "disk-spill")]
use crate::plonk::mmap_pk::{spill_with_transform, MmappedPolys};
use crate::poly::{polynomial_views, Coeff, ExtendedLagrangeCoeff, Polynomial, PolynomialView};

/// Either heap-resident cosets or cosets spilled to a mapped tempfile.
///
/// Both arms hand out basis-typed, read-only views, so storage choice remains
/// invisible to the evaluator.
pub(crate) enum Cosets<F> {
    /// The default: built in parallel and held in memory.
    Heap(Vec<Polynomial<F, ExtendedLagrangeCoeff>>),
    /// Streamed to a tempfile and mapped back, one polynomial at a time.
    #[cfg(feature = "disk-spill")]
    Spilled(MmappedPolys<F, ExtendedLagrangeCoeff>),
}

impl<F> std::fmt::Debug for Cosets<F> {
    /// Deliberately does not print the values, matching `MmappedPolys`: for the
    /// spilled arm that would fault the whole file into RAM, defeating the
    /// point of having spilled it.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Cosets::Heap(v) => f.debug_struct("Cosets::Heap").field("n_polys", &v.len()).finish(),
            #[cfg(feature = "disk-spill")]
            Cosets::Spilled(m) => f.debug_struct("Cosets::Spilled").field("inner", m).finish(),
        }
    }
}

impl<F> Cosets<F> {
    /// Borrow as read-only views, whichever arm this is.
    pub(crate) fn views(&self) -> Vec<PolynomialView<'_, F, ExtendedLagrangeCoeff>> {
        match self {
            Cosets::Heap(v) => polynomial_views(v),
            #[cfg(feature = "disk-spill")]
            Cosets::Spilled(m) => m.views(),
        }
    }
}

/// Whether the process policy spills cosets for a circuit of size `k`.
///
/// The decision itself lives on [`ProverConfig`](crate::config::ProverConfig)
/// and is pure; this only supplies the process-wide policy. The floor exists
/// because spilling is a loss at small `k` — the file write and page faults
/// cost more than the heap the small cosets would have occupied — so it earns
/// its keep only once the cosets approach the memory ceiling.
pub(crate) fn should_spill_cosets(k: u32) -> bool {
    crate::config::ProverConfig::process().spills_cosets_at(k)
}

/// Whether [`build_cosets`] will *actually* spill: the environment gate **and**
/// the capability to honour it.
///
/// [`should_spill_cosets`] answers only "was spilling asked for". Without the
/// `disk-spill` feature the answer can be yes while `build_cosets` still takes
/// the heap arm, so a caller emitting phase markers off the bare env gate would
/// report a spill that never happened. Callers wanting to narrate the choice
/// should ask this instead — and it keeps the `cfg` here rather than at every
/// call site.
pub(crate) fn will_spill(k: u32) -> bool {
    #[cfg(feature = "disk-spill")]
    {
        should_spill_cosets(k)
    }
    #[cfg(not(feature = "disk-spill"))]
    {
        let _ = k;
        false
    }
}

/// Build extended-domain cosets from `polys`, spilling to disk when
/// [`should_spill_cosets`] says so and the spill succeeds.
///
/// A spill failure is **not** fatal: it falls back to the heap path and the
/// proof is still produced. Running out of tempfile space should degrade to the
/// behaviour we had before this optimisation existed, not abort a proof.
pub(crate) fn build_cosets<F, P, D>(polys: &[P], k: u32, to_extended: D) -> Cosets<F>
where
    F: Copy + Send + Sync,
    P: crate::poly::PolynomialRead<F, Basis = Coeff> + Sync,
    D: for<'a> Fn(PolynomialView<'a, F, Coeff>) -> Polynomial<F, ExtendedLagrangeCoeff>
        + Send
        + Sync,
{
    use rayon::prelude::*;

    #[cfg(feature = "disk-spill")]
    if should_spill_cosets(k) && !polys.is_empty() {
        match spill_with_transform(polys, &to_extended) {
            Ok(m) => return Cosets::Spilled(m),
            Err(error) => {
                tracing::warn!(%error, k, "coset spill failed; falling back to heap");
                #[cfg(feature = "bench-internal")]
                if matches!(
                    std::env::var("MIDNIGHT_BENCH_REQUIRE_SPILL").as_deref(),
                    Ok("1") | Ok("true")
                ) {
                    panic!("required benchmark coset spill failed at k={k}: {error}");
                }
            }
        }
    }
    // Without `disk-spill` the gate is dead weight; keep the parameter so the
    // signature does not change with the feature.
    #[cfg(not(feature = "disk-spill"))]
    let _ = (k, should_spill_cosets(k));
    // The heap arm stays parallel. Spilling is sequential by construction —
    // streaming one polynomial at a time is what keeps peak heap at ~1 poly —
    // so the two arms trade throughput against memory, and the default path
    // must not quietly lose the parallelism it had before this existed.
    Cosets::Heap(
        polys
            .par_iter()
            .map(|poly| to_extended(PolynomialView::new(poly.values())))
            .collect(),
    )
}

#[cfg(test)]
mod test {
    // Every test in this module is `disk-spill`-only — the heap arm is
    // exercised by the prover's own tests — so the imports are gated with
    // them. Left ungated, `use super::*` is an unused import in the default
    // build, which `-D warnings` rejects.
    #[cfg(feature = "disk-spill")]
    use midnight_curves::Fq as Fp;

    #[cfg(feature = "disk-spill")]
    use super::*;
    #[cfg(feature = "disk-spill")]
    use crate::plonk::mmap_pk::spill_with_transform;

    #[test]
    #[cfg(feature = "disk-spill")]
    fn spilled_and_heap_cosets_agree() {
        // The property that matters: turning the optimisation on must not
        // change the answer. A spill producing different cosets would still
        // pass the rest of the suite, because nothing else compares the arms.
        let polys: Vec<Polynomial<Fp, Coeff>> = (0..3)
            .map(|i| Polynomial {
                values: (0..8).map(|j| Fp::from((i * 8 + j) as u64)).collect(),
                _marker: std::marker::PhantomData,
            })
            .collect();
        let widen = |p: PolynomialView<'_, Fp, Coeff>| Polynomial::<Fp, ExtendedLagrangeCoeff> {
            values: p.to_vec(),
            _marker: std::marker::PhantomData,
        };

        // Both arms are constructed directly. Going through `build_cosets`
        // would make the test depend on process-wide environment another test
        // - or the caller's shell - may have set, which is exactly what
        // splitting `spill_decision` out was meant to avoid.
        let heap = Cosets::Heap(polynomial_views(&polys).into_iter().map(widen).collect());
        let spilled = Cosets::Spilled(spill_with_transform(&polys, widen).unwrap());

        let heap = heap.views();
        let spilled = spilled.views();
        assert_eq!(heap.len(), spilled.len());
        for (h, sp) in heap.iter().zip(&spilled) {
            assert_eq!(&h[..], &sp[..], "spilled cosets must equal heap cosets");
        }
    }

    // The gate itself is now `ProverConfig::spills_cosets_at`, and
    // `config::test::the_spill_gate_needs_both_the_switch_and_the_floor`
    // asserts the same property against the type that owns it. Keeping a
    // second copy here would be two tests of one rule, drifting apart.
}
