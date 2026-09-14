//! Prover memory policy: how much RAM the prover is allowed to use, and what
//! it may spend instead.
//!
//! `docs/memory-profile.md` is the rationale — the problem, the measurements
//! including the unflattering ones, the alternatives rejected, and what a
//! reviewer should check. Read that before deciding which policy to set; these
//! settings cost time, and on a host with RAM to spare they are a straight
//! loss.
//!
//! # Why this is a type and not five `env::var` calls
//!
//! The optimisations in this crate trade memory for CPU and disk. Which trade
//! is right depends on the host — a server with RAM to spare wants none of it;
//! a phone at k=20 needs all of it — so the choice has to be expressible.
//!
//! It used to be expressed as five environment variables read at the points
//! that needed them. That has three costs, and they are the reasons this
//! module exists rather than a matter of taste:
//!
//! 1. **Process-global.** Two provers in one process cannot differ. A server
//!    that wants a small circuit on the heap and a large one spilled has no way
//!    to say so.
//! 2. **Invisible.** Nothing in any signature says the prover's behaviour
//!    depends on the environment, so a caller cannot discover the knobs, and a
//!    reviewer cannot see them.
//! 3. **Hostile to testing.** Setting an environment variable in a test is
//!    `unsafe` under Rust 2024 and races every other test in the binary. The
//!    coset gate already had to be split in two — `spill_decision` apart from
//!    `should_spill_cosets` — purely so the logic could be reached without
//!    touching the environment. That split is a symptom.
//!
//! [`ProverConfig`] is plain data. Every decision is a method on it, so the
//! decisions are directly testable. Reading the environment happens in exactly
//! one function, [`ProverConfig::from_env`], at the process boundary.
//!
//! # Compatibility
//!
//! The variables still work and mean exactly what they meant before. They are
//! now parsed once, in one place, instead of at six call sites.

use std::{path::PathBuf, sync::OnceLock};

/// Default `k` at or above which coset spilling engages, when enabled.
///
/// Spilling is a loss below this: the file write and page faults cost more
/// than the heap the small cosets would have occupied. Measured at roughly
/// 2× slower at k=14.
pub const DEFAULT_COSET_SPILL_FLOOR_K: u32 = 18;

/// Default chunk size for the Pippenger fallback, as a log2.
pub const DEFAULT_MSM_CHUNK_LOG2: u32 = 18;

/// How the prover trades memory against CPU and disk.
///
/// The default is [`ProverConfig::heap`] — every optimisation off, which is
/// upstream's behaviour. Nothing here changes what a proof *is*; these settings
/// only change where the intermediate values live. If turning one on changes a
/// proof, that is a bug, and `plonk::cosets`'s agreement test exists to catch
/// it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProverConfig {
    /// Move the prover key into a mapped file rather than holding it on the
    /// heap. Roughly neutral on time; the win is that the pages become
    /// file-backed and therefore evictable.
    pub map_prover_key: bool,

    /// Spill extended-domain cosets to a mapped tempfile at or above
    /// [`coset_spill_floor_k`](Self::coset_spill_floor_k).
    ///
    /// This is the setting that actually moves peak memory at high `k`, and
    /// it is also the one that costs the most time — around 47% at k=19 and
    /// 32% at k=20 on an unconstrained host. Enable it when memory is the
    /// binding constraint, not for speed; there is none.
    pub spill_cosets: bool,

    /// The `k` at or above which [`spill_cosets`](Self::spill_cosets) engages.
    pub coset_spill_floor_k: u32,

    /// Directory for spill tempfiles. `None` uses the OS default.
    ///
    /// Worth setting where the default temp partition is too small for the
    /// working set — k=20 needs on the order of 10 GiB.
    pub spill_dir: Option<PathBuf>,

    /// Chunk size for the Pippenger fallback, as a log2. A large value
    /// (say 32) disables chunking.
    pub msm_chunk_log2: u32,
}

impl Default for ProverConfig {
    fn default() -> Self {
        Self::heap()
    }
}

impl ProverConfig {
    /// Everything on the heap: upstream's behaviour, and the right choice
    /// wherever RAM is not the binding constraint.
    pub fn heap() -> Self {
        Self {
            map_prover_key: false,
            spill_cosets: false,
            coset_spill_floor_k: DEFAULT_COSET_SPILL_FLOOR_K,
            spill_dir: None,
            msm_chunk_log2: DEFAULT_MSM_CHUNK_LOG2,
        }
    }

    /// Map the prover key; leave cosets on the heap.
    pub fn mapped_key() -> Self {
        Self {
            map_prover_key: true,
            ..Self::heap()
        }
    }

    /// Map the prover key and spill cosets above the default floor.
    ///
    /// The profile for a memory-constrained device at high `k`. Needs
    /// temporary storage — see [`spill_dir`](Self::spill_dir).
    pub fn mapped_key_and_cosets() -> Self {
        Self {
            map_prover_key: true,
            spill_cosets: true,
            ..Self::heap()
        }
    }

    /// Read the policy from the environment.
    ///
    /// The only place this crate reads *prover policy* from the environment.
    /// (`MIDNIGHT_BENCH_REQUIRE_SPILL`, under the `bench-internal` feature, is
    /// a benchmark-harness assertion rather than a policy — it turns a silent
    /// fallback to the heap into a panic so a benchmark cannot measure the
    /// wrong arm — and is deliberately not a field here.) Unset or
    /// unparseable values fall back to the defaults rather than failing: a
    /// malformed tuning knob should not stop a prover that would otherwise
    /// work.
    ///
    /// | variable | field |
    /// |---|---|
    /// | `MIDNIGHT_SPILL_PK` | [`map_prover_key`](Self::map_prover_key) |
    /// | `MIDNIGHT_SPILL_COSETS` | [`spill_cosets`](Self::spill_cosets) |
    /// | `MIDNIGHT_SPILL_FLOOR_K` | [`coset_spill_floor_k`](Self::coset_spill_floor_k) |
    /// | `MIDNIGHT_SPILL_DIR` | [`spill_dir`](Self::spill_dir) |
    /// | `MIDNIGHT_MSM_CHUNK_LOG2` | [`msm_chunk_log2`](Self::msm_chunk_log2) |
    pub fn from_env() -> Self {
        Self {
            map_prover_key: env_flag("MIDNIGHT_SPILL_PK"),
            spill_cosets: env_flag("MIDNIGHT_SPILL_COSETS"),
            coset_spill_floor_k: env_parse("MIDNIGHT_SPILL_FLOOR_K")
                .unwrap_or(DEFAULT_COSET_SPILL_FLOOR_K),
            spill_dir: std::env::var("MIDNIGHT_SPILL_DIR")
                .ok()
                .filter(|d| !d.is_empty())
                .map(PathBuf::from),
            msm_chunk_log2: env_parse("MIDNIGHT_MSM_CHUNK_LOG2").unwrap_or(DEFAULT_MSM_CHUNK_LOG2),
        }
    }

    /// The process-wide policy, read from the environment on first use.
    ///
    /// A stepping stone, not the destination. It exists so the call sites can
    /// stop reading the environment themselves without every prover entry
    /// point growing a parameter in the same change. Threading a config
    /// through the prover is what finally allows two provers in one process to
    /// differ — see the module docs.
    pub fn process() -> &'static ProverConfig {
        static PROCESS: OnceLock<ProverConfig> = OnceLock::new();
        PROCESS.get_or_init(ProverConfig::from_env)
    }

    /// Whether cosets spill for a circuit of size `k` under this policy.
    ///
    /// Pure: no environment, no globals. This is the whole point — the gate
    /// can be tested directly, at every `k`, without the `unsafe` and the
    /// cross-test races that setting a variable would bring.
    pub fn spills_cosets_at(&self, k: u32) -> bool {
        self.spill_cosets && k >= self.coset_spill_floor_k
    }

    /// Chunk size for the Pippenger fallback.
    pub fn msm_chunk(&self) -> usize {
        1usize << self.msm_chunk_log2
    }
}

/// `1` or `true`; anything else, including unset, is false.
fn env_flag(name: &str) -> bool {
    matches!(std::env::var(name).as_deref(), Ok("1") | Ok("true"))
}

fn env_parse<T: std::str::FromStr>(name: &str) -> Option<T> {
    std::env::var(name).ok()?.parse().ok()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn heap_is_the_default_and_turns_nothing_on() {
        let c = ProverConfig::default();
        assert_eq!(c, ProverConfig::heap());
        assert!(!c.map_prover_key);
        assert!(!c.spill_cosets);
        // Off everywhere, including far above the floor.
        assert!(!c.spills_cosets_at(30));
    }

    #[test]
    fn the_spill_gate_needs_both_the_switch_and_the_floor() {
        // The property the old `spill_decision` split existed to make
        // testable, now reachable without touching the environment at all.
        let c = ProverConfig {
            spill_cosets: true,
            coset_spill_floor_k: 18,
            ..ProverConfig::heap()
        };
        assert!(!c.spills_cosets_at(17), "below the floor must not spill");
        assert!(c.spills_cosets_at(18), "at the floor must spill");
        assert!(c.spills_cosets_at(19), "above the floor must spill");

        let off = ProverConfig {
            spill_cosets: false,
            ..c.clone()
        };
        assert!(
            !off.spills_cosets_at(19),
            "the floor alone must not enable spilling"
        );
    }

    #[test]
    fn the_four_combinations_are_all_expressible() {
        // The environment variables were independent, so collapsing them into
        // a linear Heap -> MappedKey -> MappedKeyAndCosets enum would have
        // silently dropped "cosets spilled, key on the heap" — a combination
        // someone setting only MIDNIGHT_SPILL_COSETS has today.
        let cosets_only = ProverConfig {
            spill_cosets: true,
            ..ProverConfig::heap()
        };
        assert!(!cosets_only.map_prover_key);
        assert!(cosets_only.spills_cosets_at(20));

        let both = ProverConfig::mapped_key_and_cosets();
        assert!(both.map_prover_key && both.spills_cosets_at(20));

        let key_only = ProverConfig::mapped_key();
        assert!(key_only.map_prover_key && !key_only.spills_cosets_at(20));
    }

    #[test]
    fn msm_chunk_is_two_to_the_log() {
        assert_eq!(
            ProverConfig::heap().msm_chunk(),
            1 << DEFAULT_MSM_CHUNK_LOG2
        );
        let c = ProverConfig {
            msm_chunk_log2: 3,
            ..ProverConfig::heap()
        };
        assert_eq!(c.msm_chunk(), 8);
    }

    #[test]
    fn the_process_policy_is_stable_within_a_run() {
        // Read-only: setting a variable here would be `unsafe` under Rust 2024
        // and would race every other test in this binary. Asserting the cache
        // holds is what can be checked without that.
        assert_eq!(ProverConfig::process(), ProverConfig::process());
    }
}
