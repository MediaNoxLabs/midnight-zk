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
//! that needed them, plus one process-wide cancel flag. That had three costs,
//! and they are the reasons this module exists rather than a matter of taste:
//!
//! 1. **Process-global.** Two provers in one process could not differ. A server
//!    that wants a small circuit on the heap and a large one spilled had no way
//!    to say so — and cancelling one request's proof cancelled every proof in
//!    the process, key generation included.
//! 2. **Invisible.** Nothing in any signature said the prover's behaviour
//!    depended on the environment, so a caller could not discover the knobs,
//!    and a reviewer could not see them.
//! 3. **Hostile to testing.** Setting an environment variable in a test is
//!    `unsafe` under Rust 2024 and races every other test in the binary.
//!
//! [`ProverConfig`] is plain data: every memory decision is a method on it, so
//! the decisions are directly testable. [`ProverContext`] carries a config and
//! an optional [`CancelToken`] through key loading
//! (`ProvingKey::read_with_policy`) and proof creation
//! (`plonk::create_proof_with`), borrowed, so two requests in one process hold
//! two contexts and neither can see the other. Reading the environment happens
//! in exactly one function, [`ProverConfig::from_env`], at the process
//! boundary.
//!
//! One knob stays process-wide: the MSM chunk size. The commitment trait's
//! `commit(params, poly)` has no per-call seam, and widening that public trait
//! is a larger ask than the knob deserves. It is validated before use instead.
//!
//! # Compatibility
//!
//! The variables still work and mean exactly what they meant before; the
//! context-free entry points (`create_proof`, `ProvingKey::read`) run under
//! [`ProverContext::process`], which is the environment's policy plus the
//! legacy `MIDNIGHT_CANCEL` flag. The one visible change is that a set flag
//! yields `Error::Cancelled` rather than a panic.

use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, OnceLock,
    },
};

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
            msm_chunk_log2: match env_parse::<u32>("MIDNIGHT_MSM_CHUNK_LOG2") {
                Some(log2) if log2 < usize::BITS => log2,
                Some(log2) => {
                    // Not silently: an operator who typed 64 meant something,
                    // and "chunking disabled" is the closest legal reading.
                    tracing::warn!(
                        log2,
                        max = usize::BITS - 1,
                        "MIDNIGHT_MSM_CHUNK_LOG2 is out of range; chunking disabled"
                    );
                    usize::BITS - 1
                }
                None => DEFAULT_MSM_CHUNK_LOG2,
            },
        }
    }

    /// The process-wide policy, read from the environment on first use.
    ///
    /// This is the compatibility bridge behind [`ProverContext::process`]:
    /// what a caller gets when it uses an entry point that takes no context.
    /// It is read once and frozen for the process, so a host that sets the
    /// variables must do so before the first proof. Callers that want per-proof
    /// policy build a [`ProverConfig`] directly and pass a [`ProverContext`].
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
    ///
    /// `msm_chunk_log2` is clamped to `usize::BITS - 1` before the shift, so
    /// an out-of-range value — `64` from the environment, or the documented
    /// "disable" value `32` on a 32-bit target — disables chunking rather than
    /// overflowing. `1 << 31` on 32-bit or `1 << 63` on 64-bit is larger than
    /// any MSM, which is what "disabled" means here anyway.
    pub fn msm_chunk(&self) -> usize {
        1usize << self.msm_chunk_log2.min(usize::BITS - 1)
    }
}

/// A request-scoped cancellation flag.
///
/// One token belongs to one proof. Setting it stops *that* proof at its next
/// phase checkpoint with [`Error::Cancelled`](crate::plonk::Error::Cancelled);
/// it cannot reach a proof holding a different token, and it cannot reach key
/// generation, which checks no token at all. Clone it to hand the caller a
/// handle while the prover holds the other — both see the same flag.
///
/// The prover only ever *reads* the token, at phase boundaries — around 30 per
/// proof — so cancellation lands within one phase, which is the same
/// granularity the process-wide flag offered, without the process-wide part.
#[derive(Clone, Debug)]
pub struct CancelToken(TokenFlag);

#[derive(Clone, Debug)]
enum TokenFlag {
    /// A flag this token owns, shared only with its clones.
    Owned(Arc<AtomicBool>),
    /// The process-wide flag behind [`ProverContext::process`], kept so the
    /// pre-existing host protocol — set a static, expect the prover to stop —
    /// keeps working while callers move to per-request tokens.
    Shared(&'static AtomicBool),
}

impl CancelToken {
    /// A fresh token, not cancelled.
    pub fn new() -> Self {
        Self(TokenFlag::Owned(Arc::new(AtomicBool::new(false))))
    }

    /// A token over a process-wide flag. This is how the legacy
    /// `MIDNIGHT_CANCEL` static participates; new callers want
    /// [`new`](Self::new).
    pub(crate) const fn shared(flag: &'static AtomicBool) -> Self {
        Self(TokenFlag::Shared(flag))
    }

    /// Ask the proof holding this token to stop at its next checkpoint.
    pub fn cancel(&self) {
        self.flag().store(true, Ordering::Release);
    }

    /// Whether [`cancel`](Self::cancel) has been called.
    pub fn is_cancelled(&self) -> bool {
        self.flag().load(Ordering::Acquire)
    }

    fn flag(&self) -> &AtomicBool {
        match &self.0 {
            TokenFlag::Owned(a) => a,
            TokenFlag::Shared(s) => s,
        }
    }
}

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

/// Everything a single proof needs to know about *how* to run, as distinct
/// from *what* to prove: the memory policy and an optional cancellation token.
///
/// This is the value that makes two provers in one process able to differ.
/// It is borrowed through key loading and proof creation, so it costs nothing
/// per call, and every decision the prover makes about memory or cancellation
/// is answered from it rather than from process state.
///
/// [`ProverContext::process`] is the compatibility bridge: the same policy
/// the environment variables have always expressed, plus the process-wide
/// cancel flag. Entry points that take no context use it, so existing callers
/// see exactly the behaviour they had — except that cancellation is now an
/// error, not a panic.
#[derive(Clone, Debug, Default)]
pub struct ProverContext {
    /// How this proof trades memory for CPU and disk.
    pub config: ProverConfig,
    /// Set to let the caller stop this proof between phases. `None` means the
    /// proof cannot be cancelled, which is the right default for a library
    /// call that owns the whole computation.
    pub cancel: Option<CancelToken>,
}

impl ProverContext {
    /// A context with the given policy and no cancellation.
    pub fn new(config: ProverConfig) -> Self {
        Self {
            config,
            cancel: None,
        }
    }

    /// Attach a cancellation token.
    pub fn with_cancel(mut self, token: CancelToken) -> Self {
        self.cancel = Some(token);
        self
    }

    /// Whether cancellation has been requested for this proof.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(CancelToken::is_cancelled)
    }

    /// The process-wide context: [`ProverConfig::process`] plus the legacy
    /// `MIDNIGHT_CANCEL` flag as its token.
    ///
    /// This is what every context-free entry point uses, so callers that
    /// never heard of a context keep the behaviour the environment variables
    /// and the static flag gave them. The one deliberate change: a set flag
    /// now yields [`Error::Cancelled`](crate::plonk::Error::Cancelled)
    /// instead of a panic. Its `Display` still carries the old sentinel.
    pub fn process() -> Self {
        Self {
            config: ProverConfig::process().clone(),
            cancel: Some(CancelToken::shared(&crate::plonk::MIDNIGHT_CANCEL)),
        }
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

    #[test]
    fn a_token_cancels_its_holder_and_nothing_else() {
        let a = CancelToken::new();
        let b = CancelToken::new();
        let a_handle = a.clone();
        assert!(!a.is_cancelled() && !b.is_cancelled());

        a_handle.cancel();
        assert!(a.is_cancelled(), "the clone shares the flag");
        assert!(!b.is_cancelled(), "an unrelated token is untouched");

        let ctx_a = ProverContext::new(ProverConfig::heap()).with_cancel(a);
        let ctx_b = ProverContext::new(ProverConfig::heap()).with_cancel(b);
        let ctx_none = ProverContext::new(ProverConfig::heap());
        assert!(ctx_a.is_cancelled());
        assert!(!ctx_b.is_cancelled());
        assert!(!ctx_none.is_cancelled(), "no token means never cancelled");
    }

    #[test]
    fn msm_chunk_never_overflows_the_shift() {
        // F-025: the documented "disable" value 32 overflows a 32-bit usize,
        // and 64 was accepted from the environment on 64-bit. Both now clamp
        // to the largest representable power of two, which is "disabled".
        for log2 in [usize::BITS - 1, usize::BITS, 64, 200, u32::MAX] {
            let c = ProverConfig {
                msm_chunk_log2: log2,
                ..ProverConfig::heap()
            };
            assert_eq!(c.msm_chunk(), 1usize << (usize::BITS - 1), "log2={log2}");
        }
    }

    #[test]
    fn the_cancelled_error_still_carries_the_sentinel() {
        // Hosts that predate the typed error recognised cancellation by this
        // substring in a panic message. It must survive in the error's text.
        let e = crate::plonk::Error::Cancelled {
            phase: "trace.parse_advices.start",
        };
        let text = e.to_string();
        assert!(
            text.contains(crate::plonk::MIDNIGHT_CANCEL_SENTINEL),
            "{text}"
        );
        assert!(text.contains("trace.parse_advices.start"), "{text}");
    }
}
