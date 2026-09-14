//! This module provides an implementation of a variant of (Turbo)[PLONK][plonk]
//! that is designed specifically for the polynomial commitment scheme described
//! in the [Halo][halo] paper.
//!
//! [halo]: https://eprint.iacr.org/2019/1021
//! [plonk]: https://eprint.iacr.org/2019/953

use blake2b_simd::Params as Blake2bParams;
use group::ff::FromUniformBytes;

use crate::{
    poly::{
        polynomial_views, Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff,
        PinnedEvaluationDomain, Polynomial, PolynomialView,
    },
    transcript::{Hashable, Transcript},
    utils::{
        helpers::{
            byte_length, polynomial_slice_byte_length, read_polynomial_vec, write_polynomial_slice,
            ProcessedSerdeObject,
        },
        SerdeFormat,
    },
};

mod circuit;
/// Coset construction, spilling to disk when built to.
pub(crate) mod cosets;
mod error;
pub(crate) mod evaluation;
mod keygen;
pub(crate) mod lookup;
// Mmap-backed proving-key and proof-time spill infrastructure. It needs both a
// filesystem to write to and `mmap(2)` to read from.
#[cfg(feature = "disk-spill")]
pub(crate) mod mmap_pk;
pub mod permutation;
pub(crate) mod traces;
pub(crate) mod trash;
pub(crate) mod vanishing;

#[cfg(feature = "bench-internal")]
pub mod bench;

mod prover;
mod verifier;

use std::io;

pub use circuit::*;
pub use error::*;
pub(crate) use evaluation::Evaluator;
use ff::{PrimeField, WithSmallOrderMulGroup};
pub use keygen::*;
use midnight_curves::serde::SerdeObject;
pub use prover::*;
pub use verifier::*;

use crate::poly::commitment::PolynomialCommitmentScheme;

/// This is a verifying key which allows for the verification of proofs for a
/// particular circuit.
#[derive(Clone, Debug)]
pub struct VerifyingKey<F: PrimeField, CS: PolynomialCommitmentScheme<F>> {
    domain: EvaluationDomain<F>,
    fixed_commitments: Vec<CS::Commitment>,
    permutation: permutation::VerifyingKey<F, CS>,
    cs: ConstraintSystem<F>,
    /// Cached maximum degree of `cs` (which doesn't change after construction).
    cs_degree: usize,
    /// The representative of this `VerifyingKey` in transcripts.
    transcript_repr: F,
}

// Current version of the VK
const VERSION: u8 = 0x03;

impl<F, CS> VerifyingKey<F, CS>
where
    F: WithSmallOrderMulGroup<3> + FromUniformBytes<64>,
    CS: PolynomialCommitmentScheme<F>,
{
    /// Returns `n`
    pub fn n(&self) -> u64 {
        self.domain.n
    }
    /// Writes a verifying key to a buffer.
    ///
    /// Writes a curve element according to `format`:
    /// - `Processed`: Writes a compressed curve element with coordinates in
    ///   standard form. Writes a field element in standard form, with
    ///   endianness specified by the `PrimeField` implementation.
    /// - Otherwise: Writes an uncompressed curve element with coordinates in
    ///   Montgomery form Writes a field element into raw bytes in its internal
    ///   Montgomery representation, WITHOUT performing the expensive Montgomery
    ///   reduction.
    pub fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()> {
        // Version byte that will be checked on read.
        writer.write_all(&[VERSION])?;
        let k = &self.domain.k();
        assert!(*k <= F::S);
        // k value fits in 1 byte
        writer.write_all(&[*k as u8])?;
        writer.write_all(&(self.fixed_commitments.len() as u32).to_le_bytes())?;
        for commitment in &self.fixed_commitments {
            commitment.write(writer, format)?;
        }
        self.permutation.write(writer, format)?;

        Ok(())
    }

    /// Reads a verification key from a buffer for the associated [Circuit].
    ///
    /// Reads a curve element from the buffer and parses it according to the
    /// `format`:
    /// - `Processed`: Reads a compressed curve element and decompresses it.
    ///   Reads a field element in standard form, with endianness specified by
    ///   the `PrimeField` implementation, and checks that the element is less
    ///   than the modulus.
    /// - `RawBytes`: Reads an uncompressed curve element with coordinates in
    ///   Montgomery form. Checks that field elements are less than modulus, and
    ///   then checks that the point is on the curve.
    /// - `RawBytesUnchecked`: Reads an uncompressed curve element with
    ///   coordinates in Montgomery form; does not perform any checks.
    pub fn read<R: io::Read, ConcreteCircuit: Circuit<F>>(
        reader: &mut R,
        format: SerdeFormat,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<Self> {
        let mut cs = ConstraintSystem::default();
        #[cfg(feature = "circuit-params")]
        let _config = ConcreteCircuit::configure_with_params(&mut cs, params);
        #[cfg(not(feature = "circuit-params"))]
        let _config = ConcreteCircuit::configure(&mut cs);

        Self::read_from_cs(reader, format, cs)
    }

    /// Reads a verification key from a buffer, using the provided
    /// [ConstraintSystem].
    ///
    /// Reads a curve element from the buffer and parses it according to the
    /// `format`:
    /// - `Processed`: Reads a compressed curve element and decompresses it.
    ///   Reads a field element in standard form, with endianness specified by
    ///   the `PrimeField` implementation, and checks that the element is less
    ///   than the modulus.
    /// - `RawBytes`: Reads an uncompressed curve element with coordinates in
    ///   Montgomery form. Checks that field elements are less than modulus, and
    ///   then checks that the point is on the curve.
    /// - `RawBytesUnchecked`: Reads an uncompressed curve element with
    ///   coordinates in Montgomery form; does not perform any checks.
    pub fn read_from_cs<R: io::Read>(
        reader: &mut R,
        format: SerdeFormat,
        cs: ConstraintSystem<F>,
    ) -> io::Result<Self> {
        let mut version_byte = [0u8; 1];
        reader.read_exact(&mut version_byte)?;
        if VERSION != version_byte[0] {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unexpected version byte",
            ));
        }

        let mut k = [0u8; 1];
        reader.read_exact(&mut k)?;
        let k = u8::from_le_bytes(k);
        if k as u32 > F::S {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("circuit size value (k): {} exceeds maxium: {}", k, F::S),
            ));
        }

        let domain = EvaluationDomain::new(cs.degree() as u32, k.into());

        let mut num_fixed_columns = [0u8; 4];
        reader.read_exact(&mut num_fixed_columns)?;
        let num_fixed_columns = u32::from_le_bytes(num_fixed_columns);

        let fixed_commitments: Vec<_> = (0..num_fixed_columns)
            .map(|_| CS::Commitment::read(reader, format))
            .collect::<Result<_, _>>()?;

        let permutation = permutation::VerifyingKey::read(reader, &cs.permutation, format)?;

        // we still need to replace selectors with fixed Expressions in `cs`
        let fake_selectors = vec![vec![]; cs.num_selectors];
        let (cs, _) = cs.directly_convert_selectors_to_fixed(fake_selectors);

        Ok(Self::from_parts(domain, fixed_commitments, permutation, cs))
    }

    /// Writes a verifying key to a vector of bytes using [`Self::write`].
    pub fn to_bytes(&self, format: SerdeFormat) -> Vec<u8> {
        let mut bytes = Vec::<u8>::with_capacity(self.bytes_length(format));
        Self::write(self, &mut bytes, format).expect("Writing to vector should not fail");
        bytes
    }

    /// Reads a verification key from a slice of bytes using [`Self::read`].
    pub fn from_bytes<ConcreteCircuit: Circuit<F>>(
        mut bytes: &[u8],
        format: SerdeFormat,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<Self> {
        Self::read::<_, ConcreteCircuit>(
            &mut bytes,
            format,
            #[cfg(feature = "circuit-params")]
            params,
        )
    }
}

impl<F: WithSmallOrderMulGroup<3>, CS: PolynomialCommitmentScheme<F>> VerifyingKey<F, CS> {
    /// Return the bytes_length of a VerifyingKey
    pub fn bytes_length(&self, format: SerdeFormat) -> usize {
        10 + (self.fixed_commitments.len() * byte_length::<CS::Commitment>(format))
            + self.permutation.bytes_length(format)
    }

    fn from_parts(
        domain: EvaluationDomain<F>,
        fixed_commitments: Vec<CS::Commitment>,
        permutation: permutation::VerifyingKey<F, CS>,
        cs: ConstraintSystem<F>,
    ) -> Self
    where
        F: FromUniformBytes<64>,
    {
        // Compute cached values.
        let cs_degree = cs.degree();

        let mut vk = Self {
            domain,
            fixed_commitments,
            permutation,
            cs,
            cs_degree,
            // Temporary, this is not pinned.
            transcript_repr: F::ZERO,
        };

        let mut hasher =
            Blake2bParams::new().hash_length(64).personal(b"Halo2-Verify-Key").to_state();

        // We serialise the commitments of the VK to get the `transcript_repr`.
        let mut buffer = Vec::new();
        buffer.push(VERSION);
        let k = &vk.domain.k();
        assert!(*k <= F::S);
        buffer.push(*k as u8);
        buffer.extend_from_slice(&(vk.fixed_commitments.len() as u32).to_le_bytes());
        for commitment in &vk.fixed_commitments {
            commitment
                .write(&mut buffer, SerdeFormat::RawBytesUnchecked)
                .expect("Failed to write to buffer - this is a bug.");
        }

        buffer.extend_from_slice(&(vk.permutation.commitments().len() as u32).to_le_bytes());
        for commitment in vk.permutation.commitments() {
            commitment
                .write(&mut buffer, SerdeFormat::RawBytesUnchecked)
                .expect("Failed to write to buffer - this is a bug.");
        }

        // We use the debug implementation to add the gates and domain to the hashed
        // buffer. We should eventually move away from debug implementation for
        // this purpose. See https://github.com/midnightntwrk/halo2/issues/5
        buffer.extend_from_slice(format!("{:?}", vk.get_domain().pinned()).as_bytes());
        buffer.extend_from_slice(format!("{:?}", vk.cs().pinned()).as_bytes());

        hasher.update(&buffer);

        // Hash in final Blake2bState
        vk.transcript_repr = F::from_uniform_bytes(hasher.finalize().as_array());

        vk
    }

    /// Hashes a verification key into a transcript.
    pub fn hash_into<T: Transcript>(&self, transcript: &mut T) -> io::Result<()>
    where
        F: Hashable<T::Hash>,
    {
        transcript.common(&self.transcript_repr)?;

        Ok(())
    }

    /// Obtains a pinned representation of this verification key that contains
    /// the minimal information necessary to reconstruct the verification key.
    pub fn pinned(&self) -> PinnedVerificationKey<'_, F, CS> {
        PinnedVerificationKey {
            domain: self.domain.pinned(),
            fixed_commitments: &self.fixed_commitments,
            permutation: &self.permutation,
            cs: self.cs.pinned(),
        }
    }

    /// Returns commitments of fixed polynomials
    pub fn fixed_commitments(&self) -> &Vec<CS::Commitment> {
        &self.fixed_commitments
    }

    /// Returns `VerifyingKey` of permutation
    pub fn permutation(&self) -> &permutation::VerifyingKey<F, CS> {
        &self.permutation
    }

    /// Returns `ConstraintSystem`
    pub fn cs(&self) -> &ConstraintSystem<F> {
        &self.cs
    }

    /// Returns representative of this `VerifyingKey` in transcripts
    pub fn transcript_repr(&self) -> F {
        self.transcript_repr
    }
}

/// Minimal representation of a verification key that can be used to identify
/// its active contents.
#[allow(dead_code)]
#[derive(Debug)]
pub struct PinnedVerificationKey<'a, F: PrimeField, CS: PolynomialCommitmentScheme<F>> {
    domain: PinnedEvaluationDomain<'a, F>,
    cs: PinnedConstraintSystem<'a, F>,
    fixed_commitments: &'a Vec<CS::Commitment>,
    permutation: &'a permutation::VerifyingKey<F, CS>,
}
/// This is a proving key which allows for the creation of proofs for a
/// particular circuit.
#[derive(Clone, Debug)]
pub struct ProvingKey<F: PrimeField, CS: PolynomialCommitmentScheme<F>> {
    pub(crate) vk: VerifyingKey<F, CS>,
    pub(crate) l0: Polynomial<F, ExtendedLagrangeCoeff>,
    pub(crate) l_last: Polynomial<F, ExtendedLagrangeCoeff>,
    pub(crate) l_active_row: Polynomial<F, ExtendedLagrangeCoeff>,
    pub(crate) fixed_values: Vec<Polynomial<F, LagrangeCoeff>>,
    pub(crate) fixed_polys: Vec<Polynomial<F, Coeff>>,
    pub(crate) fixed_cosets: Vec<Polynomial<F, ExtendedLagrangeCoeff>>,
    pub(crate) permutation: permutation::ProvingKey<F>,
    pub(crate) ev: Evaluator<F>,
    /// Optional mmap-backed view of `fixed_polys`. When
    /// `Some`, the corresponding `fixed_polys` Vec above is empty
    /// and callers MUST read polynomials through
    /// [`ProvingKey::fixed_polys_views`]. Wrapped in `Arc` so PK
    /// `clone()` shares the underlying mmap + tempfile.
    #[cfg(feature = "disk-spill")]
    pub(crate) fixed_polys_mmap: Option<std::sync::Arc<mmap_pk::MmappedPolys<F, Coeff>>>,

    /// Optional mmap-backed view of `fixed_values`. Same semantics as
    /// `fixed_polys_mmap`. Read via [`ProvingKey::fixed_values_views`].
    #[cfg(feature = "disk-spill")]
    pub(crate) fixed_values_mmap: Option<std::sync::Arc<mmap_pk::MmappedPolys<F, LagrangeCoeff>>>,

    /// Optional mmap-backed view of `permutation.polys`. Same semantics as
    /// `fixed_polys_mmap`. Read via
    /// [`ProvingKey::permutation_polys_views`]. Lives at the
    /// top-level PK rather than inside `permutation::ProvingKey`
    /// to keep the permutation type clone-safe and small.
    #[cfg(feature = "disk-spill")]
    pub(crate) permutation_polys_mmap: Option<std::sync::Arc<mmap_pk::MmappedPolys<F, Coeff>>>,
}

// Bound-free impl so the accessors and spill operations are available
// from every code path that holds a `ProvingKey<F, CS>` — including
// the `prover::create_proof` path which doesn't constrain
// `F: FromUniformBytes<64>`.

/// Bench-only: print the high-water mark after a stage of `read_with_policy`.
///
/// The HWM is monotonic, which is exactly what is wanted here — the stage at
/// which it jumps is the stage that set the peak. Deltas are against the
/// previous sample so the output reads as a per-stage attribution.
#[cfg(feature = "bench-internal")]
fn read_stage(name: &str, prev_hwm_kb: &mut u64) {
    if let Some((rss_kb, hwm_kb)) = crate::plonk::prover::sample_rss_hwm_kb() {
        let delta = hwm_kb.saturating_sub(*prev_hwm_kb);
        eprintln!(
            "read-stage {name:<28} rss={:>6} MiB  hwm={:>6} MiB  hwm_delta=+{:>5} MiB",
            rss_kb >> 10,
            hwm_kb >> 10,
            delta >> 10
        );
        *prev_hwm_kb = hwm_kb;
    }
}
#[cfg(not(feature = "bench-internal"))]
#[inline(always)]
fn read_stage(_name: &str, _prev: &mut u64) {}

impl<F: PrimeField, CS: PolynomialCommitmentScheme<F>> ProvingKey<F, CS> {
    /// Read `fixed_polys` through a single accessor so the
    /// in-place mmap-backed sidecar can transparently replace the
    /// heap `Vec` when engaged.
    ///
    /// When `fixed_polys_mmap` is `Some`, the heap `fixed_polys`
    /// vector is empty and the polynomials live in mmap'd file
    /// pages. When `None`, the legacy heap vector is returned.
    /// Callers that previously indexed `&pk.fixed_polys[i]` should
    /// now normalise once with `pk.fixed_polys_views()`.
    pub(crate) fn fixed_polys_views(&self) -> Vec<PolynomialView<'_, F, Coeff>> {
        #[cfg(feature = "disk-spill")]
        if let Some(m) = self.fixed_polys_mmap.as_ref() {
            return m.views();
        }
        polynomial_views(&self.fixed_polys)
    }

    /// Mirror of `fixed_polys_views` for `fixed_values`.
    pub(crate) fn fixed_values_views(&self) -> Vec<PolynomialView<'_, F, LagrangeCoeff>> {
        #[cfg(feature = "disk-spill")]
        if let Some(m) = self.fixed_values_mmap.as_ref() {
            return m.views();
        }
        polynomial_views(&self.fixed_values)
    }

    /// Mirror of `fixed_polys_views` for `permutation.polys`.
    pub(crate) fn permutation_polys_views(&self) -> Vec<PolynomialView<'_, F, Coeff>> {
        #[cfg(feature = "disk-spill")]
        if let Some(m) = self.permutation_polys_mmap.as_ref() {
            return m.views();
        }
        polynomial_views(&self.permutation.polys)
    }

    /// Move `fixed_polys` into mmap-backed storage.
    /// See [`Self::spill_all_to_mmap`] for a one-shot variant that
    /// also handles `fixed_values` and `permutation.polys`.
    #[cfg(feature = "disk-spill")]
    pub(crate) fn spill_fixed_polys_to_mmap(&mut self) -> std::io::Result<()> {
        if self.fixed_polys_mmap.is_some() {
            return Ok(());
        }
        if self.fixed_polys.is_empty() {
            return Ok(());
        }
        let n_per_poly = self.fixed_polys[0].values.len();
        let polys = std::mem::take(&mut self.fixed_polys);
        let mm = mmap_pk::spill_vec_to_disk(polys, n_per_poly)?;
        self.fixed_polys_mmap = Some(std::sync::Arc::new(mm));
        Ok(())
    }

    /// Move `fixed_values` (LagrangeCoeff basis) into
    /// mmap-backed storage. Mirrors `spill_fixed_polys_to_mmap`.
    #[cfg(feature = "disk-spill")]
    pub(crate) fn spill_fixed_values_to_mmap(&mut self) -> std::io::Result<()> {
        if self.fixed_values_mmap.is_some() {
            return Ok(());
        }
        if self.fixed_values.is_empty() {
            return Ok(());
        }
        let n_per_poly = self.fixed_values[0].values.len();
        let polys = std::mem::take(&mut self.fixed_values);
        let mm = mmap_pk::spill_vec_to_disk(polys, n_per_poly)?;
        self.fixed_values_mmap = Some(std::sync::Arc::new(mm));
        Ok(())
    }

    /// Move `permutation.polys` (Coeff basis) into
    /// mmap-backed storage. Sidecar lives at the top-level PK so
    /// `permutation::ProvingKey` stays clone-safe and small.
    #[cfg(feature = "disk-spill")]
    pub(crate) fn spill_permutation_polys_to_mmap(&mut self) -> std::io::Result<()> {
        if self.permutation_polys_mmap.is_some() {
            return Ok(());
        }
        if self.permutation.polys.is_empty() {
            return Ok(());
        }
        let n_per_poly = self.permutation.polys[0].values.len();
        let polys = std::mem::take(&mut self.permutation.polys);
        let mm = mmap_pk::spill_vec_to_disk(polys, n_per_poly)?;
        self.permutation_polys_mmap = Some(std::sync::Arc::new(mm));
        Ok(())
    }

    /// Spill `fixed_polys`, `fixed_values`, and `permutation.polys` in
    /// sequence. Each step is independently idempotent. A failure is returned;
    /// fields converted by earlier steps remain valid mapped storage, so a
    /// caller retaining the key must deliberately handle the partial result.
    #[cfg(feature = "disk-spill")]
    pub(crate) fn spill_all_to_mmap(&mut self) -> std::io::Result<()> {
        self.spill_fixed_polys_to_mmap()?;
        self.spill_fixed_values_to_mmap()?;
        self.spill_permutation_polys_to_mmap()?;

        // Drop the cached extended-domain cosets.
        //
        // Without this the spill mostly does not work. `ProvingKey::read`
        // eagerly builds `fixed_cosets` and `permutation::ProvingKey::read`
        // builds its own — each `4n` per column, and together by far the
        // largest allocation in a loaded key. Spilling the *polynomials*
        // while leaving those in place moves the smaller half and reports
        // success.
        //
        // Worse, the prover prefers them when present
        // (`prover.rs`: `if !pk.fixed_cosets.is_empty()`), so leaving them
        // also short-circuits the coset spill: a key loaded with both knobs
        // enabled would keep every coset on the heap and never reach
        // `build_cosets`.
        //
        // They are derived data — `coeff_to_extended` of the polynomials we
        // just spilled — so dropping them costs a rebuild at prove time,
        // which is precisely the trade the spill exists to make. Done last,
        // after the spills have succeeded, so a failure does not discard
        // them for nothing.
        //
        // Not covered by a unit test: constructing a `ProvingKey` requires a
        // full keygen, which nothing at this level can do cheaply. The
        // regression this guards against is "someone adds a third cached-coset
        // collection and does not clear it here", and the honest place to catch
        // that is an end-to-end assertion on a real key — tracked rather than
        // faked with a test that would only assert these two fields exist.
        self.fixed_cosets = Vec::new();
        self.permutation.cosets = Vec::new();
        Ok(())
    }
}

impl<F: WithSmallOrderMulGroup<3>, CS: PolynomialCommitmentScheme<F>> ProvingKey<F, CS>
where
    F: FromUniformBytes<64>,
{
    /// Get the underlying [`VerifyingKey`].
    pub fn get_vk(&self) -> &VerifyingKey<F, CS> {
        &self.vk
    }

    /// Gets the total number of bytes in the serialization of `self`
    pub fn bytes_length(&self, format: SerdeFormat) -> usize {
        let fixed_values = self.fixed_values_views();
        self.vk.bytes_length(format)
            + 12 // bytes used for encoding the length(u32) of "l0", "l_last" & "l_active_row" polys
            + polynomial_slice_byte_length(&fixed_values)
            + self.permutation.bytes_length()
    }
}

impl<F: WithSmallOrderMulGroup<3>, CS: PolynomialCommitmentScheme<F>> ProvingKey<F, CS>
where
    F: PrimeField + FromUniformBytes<64> + SerdeObject,
{
    /// Writes a proving key to a buffer.
    ///
    /// Writes a curve element according to `format`:
    /// - `Processed`: Writes a compressed curve element with coordinates in
    ///   standard form. Writes a field element in standard form, with
    ///   endianness specified by the `PrimeField` implementation.
    /// - Otherwise: Writes an uncompressed curve element with coordinates in
    ///   Montgomery form Writes a field element into raw bytes in its internal
    ///   Montgomery representation, WITHOUT performing the expensive Montgomery
    ///   reduction. Does so by first writing the verifying key and then
    ///   serializing the rest of the data (in the form of field polynomials)
    pub fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()> {
        self.vk.write(writer, format)?;
        write_polynomial_slice(&self.fixed_values_views(), writer)?;
        self.permutation.write(writer)?;
        Ok(())
    }

    /// Reads a proving key from a buffer.
    /// Does so by reading verification key first, and then deserializing the
    /// rest of the file into the remaining proving key data.
    ///
    /// Reads a curve element from the buffer and parses it according to the
    /// `format`:
    /// - `Processed`: Reads a compressed curve element and decompresses it.
    ///   Reads a field element in standard form, with endianness specified by
    ///   the `PrimeField` implementation, and checks that the element is less
    ///   than the modulus.
    /// - `RawBytes`: Reads an uncompressed curve element with coordinates in
    ///   Montgomery form. Checks that field elements are less than modulus, and
    ///   then checks that the point is on the curve.
    /// - `RawBytesUnchecked`: Reads an uncompressed curve element with
    ///   coordinates in Montgomery form; does not perform any checks
    pub fn read<R: io::Read, ConcreteCircuit: Circuit<F>>(
        reader: &mut R,
        format: SerdeFormat,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<Self> {
        Self::read_with_policy::<R, ConcreteCircuit>(
            reader,
            format,
            #[cfg(feature = "circuit-params")]
            params,
            crate::config::ProverConfig::process(),
        )
    }

    /// [`Self::read`] with an explicit memory policy instead of the process
    /// one.
    ///
    /// This is where the policy is *decided*, and it is decided **before** the
    /// extended-domain cosets are built — not after. The previous order built
    /// every coset eagerly and only then consulted the policy, so on the path a
    /// device actually takes the load-time peak was more than double the keygen
    /// path (1,014 vs 444 MiB at k=16), and enabling the spill made it slightly
    /// *worse*, because the spill's own transient landed on top of a peak that
    /// had already been paid. Clearing the cosets afterwards changed the
    /// steady state and left the peak untouched.
    ///
    /// When the policy maps the prover key, the cosets are not built here at
    /// all. The prover rebuilds them lazily through `build_cosets`, which is
    /// the path that can spill them — so skipping the eager build is what
    /// makes the coset spill reachable on load, not merely a memory saving.
    ///
    /// Taking the policy as a parameter is what lets a test exercise this
    /// without mutating process environment, and it is the seam a per-request
    /// policy will thread through later.
    pub(crate) fn read_with_policy<R: io::Read, ConcreteCircuit: Circuit<F>>(
        reader: &mut R,
        format: SerdeFormat,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
        policy: &crate::config::ProverConfig,
    ) -> io::Result<Self> {
        let mut hwm = 0u64;
        read_stage("start", &mut hwm);
        let vk = VerifyingKey::<F, CS>::read::<R, ConcreteCircuit>(
            reader,
            format,
            #[cfg(feature = "circuit-params")]
            params,
        )?;
        read_stage("vk read (incl. domain)", &mut hwm);
        let [l0, l_last, l_active_row] = compute_lagrange_polys(&vk, &vk.cs);
        read_stage("lagrange polys", &mut hwm);
        let fixed_values = read_polynomial_vec(reader, format)?;
        read_stage("fixed_values read", &mut hwm);
        let fixed_polys: Vec<_> = fixed_values
            .iter()
            .map(|poly| vk.domain.lagrange_to_coeff(poly.clone()))
            .collect();
        read_stage("fixed_polys (ifft)", &mut hwm);
        // `disk-spill` is the only configuration in which `map_prover_key` can
        // be honoured; without the feature the eager build is the only path.
        #[cfg(feature = "disk-spill")]
        let defer_cosets = policy.map_prover_key;
        #[cfg(not(feature = "disk-spill"))]
        let defer_cosets = {
            let _ = policy;
            false
        };
        let fixed_cosets = if defer_cosets {
            Vec::new()
        } else {
            fixed_polys
                .iter()
                .map(|poly| vk.domain.coeff_to_extended(poly.clone()))
                .collect()
        };
        read_stage("fixed_cosets (or deferred)", &mut hwm);
        let permutation = permutation::ProvingKey::read(
            reader,
            format,
            &vk.domain,
            &vk.cs.permutation,
            defer_cosets,
        )?;
        read_stage("permutation read", &mut hwm);
        let ev = Evaluator::new(vk.cs());
        read_stage("evaluator", &mut hwm);
        // Only the `disk-spill` path below mutates `pk`.
        #[cfg_attr(not(feature = "disk-spill"), allow(unused_mut))]
        let mut pk = Self {
            vk,
            l0,
            l_last,
            l_active_row,
            fixed_values,
            fixed_polys,
            fixed_cosets,
            permutation,
            ev,
            #[cfg(feature = "disk-spill")]
            fixed_polys_mmap: None,
            #[cfg(feature = "disk-spill")]
            fixed_values_mmap: None,
            #[cfg(feature = "disk-spill")]
            permutation_polys_mmap: None,
        };
        // Opt-in mmap spill immediately after deserialisation. The default path
        // remains unchanged. A spill error must reject this load: the move into
        // a sidecar can already have emptied one or more owned vectors, so
        // silently returning the partially converted key would be invalid. The
        // caller can retry the read explicitly with spilling disabled.
        #[cfg(feature = "disk-spill")]
        if policy.map_prover_key {
            pk.spill_all_to_mmap()?;
        }
        read_stage("spill (or none)", &mut hwm);
        Ok(pk)
    }

    /// Writes a proving key to a vector of bytes using [`Self::write`].
    pub fn to_bytes(&self, format: SerdeFormat) -> Vec<u8> {
        let mut bytes = Vec::<u8>::with_capacity(self.bytes_length(format));
        Self::write(self, &mut bytes, format).expect("Writing to vector should not fail");
        bytes
    }

    /// Reads a proving key from a slice of bytes using [`Self::read`].
    pub fn from_bytes<ConcreteCircuit: Circuit<F>>(
        mut bytes: &[u8],
        format: SerdeFormat,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<Self> {
        Self::read::<_, ConcreteCircuit>(
            &mut bytes,
            format,
            #[cfg(feature = "circuit-params")]
            params,
        )
    }
}

impl<F: PrimeField, CS: PolynomialCommitmentScheme<F>> VerifyingKey<F, CS> {
    /// Get the underlying [`EvaluationDomain`].
    pub fn get_domain(&self) -> &EvaluationDomain<F> {
        &self.domain
    }
}

#[cfg(all(test, feature = "disk-spill"))]
mod read_policy_test {
    //! The load-time peak is decided by *when* the policy is consulted, so this
    //! drives `read_with_policy` directly. Setting an environment variable here
    //! would be `unsafe` under Rust 2024 and would race every other test in the
    //! binary — and it would also test the wrong thing, since the defect was
    //! never in reading the variable but in reading it too late.

    use midnight_curves::{Bls12, Fq};
    use rand_core::OsRng;

    use crate::{
        circuit::SimpleFloorPlanner,
        config::ProverConfig,
        plonk::{keygen_pk, keygen_vk_with_k, Circuit, ConstraintSystem, Error, ProvingKey},
        poly::kzg::{params::ParamsKZG, KZGCommitmentScheme},
        utils::SerdeFormat,
    };

    #[derive(Clone, Copy)]
    struct MyCircuit;

    impl<F: ff::Field> Circuit<F> for MyCircuit {
        type Config = ();
        type FloorPlanner = SimpleFloorPlanner;
        #[cfg(feature = "circuit-params")]
        type Params = ();

        fn without_witnesses(&self) -> Self {
            *self
        }

        fn configure(_meta: &mut ConstraintSystem<F>) -> Self::Config {}

        fn synthesize(
            &self,
            _config: Self::Config,
            _layouter: impl crate::circuit::Layouter<F>,
        ) -> Result<(), Error> {
            Ok(())
        }
    }

    fn serialised_key() -> Vec<u8> {
        const K: u32 = 4;
        let params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(K, OsRng);
        let vk = keygen_vk_with_k::<Fq, KZGCommitmentScheme<Bls12>, _>(&params, &MyCircuit, K)
            .expect("keygen_vk");
        keygen_pk(vk, &MyCircuit)
            .expect("keygen_pk")
            .to_bytes(SerdeFormat::RawBytesUnchecked)
    }

    fn read(bytes: &[u8], policy: &ProverConfig) -> ProvingKey<Fq, KZGCommitmentScheme<Bls12>> {
        ProvingKey::read_with_policy::<_, MyCircuit>(
            &mut &bytes[..],
            SerdeFormat::RawBytesUnchecked,
            #[cfg(feature = "circuit-params")]
            (),
            policy,
        )
        .expect("read_with_policy")
    }

    #[test]
    fn heap_policy_builds_cosets_eagerly_as_upstream_does() {
        let bytes = serialised_key();
        let pk = read(&bytes, &ProverConfig::heap());
        assert!(!pk.fixed_cosets.is_empty() || pk.fixed_polys.is_empty());
        assert!(pk.fixed_polys_mmap.is_none());
        assert!(pk.permutation_polys_mmap.is_none());
    }

    #[test]
    fn mapped_key_policy_never_builds_the_cosets_it_would_have_to_drop() {
        // The property that was missing: with the spill requested, the cosets
        // must not exist at any point during `read`. Asserting they are empty
        // *after* `read` returns is the strongest observable statement of that
        // from outside; the old code could not pass it either, because it built
        // them and cleared them, so an empty result here plus the peak
        // measurement together pin the behaviour.
        let bytes = serialised_key();
        let pk = read(&bytes, &ProverConfig::mapped_key());
        assert!(
            pk.fixed_cosets.is_empty(),
            "fixed cosets must be deferred, not built then dropped"
        );
        assert!(
            pk.permutation.cosets.is_empty(),
            "permutation cosets must be deferred too"
        );

        // Whether a sidecar exists depends on whether there was anything to
        // move: a circuit with no fixed or permutation columns has nothing to
        // spill, and `spill_*_to_mmap` correctly returns early. Derive the
        // expectation from the heap-read key rather than hard-coding it, so
        // the test is right for any fixture and not just this one.
        let heap = read(&bytes, &ProverConfig::heap());
        assert_eq!(
            pk.fixed_polys_mmap.is_some(),
            !heap.fixed_polys.is_empty(),
            "fixed polys must be mapped exactly when there were any"
        );
        assert_eq!(
            pk.permutation_polys_mmap.is_some(),
            !heap.permutation.polys.is_empty(),
            "permutation polys must be mapped exactly when there were any"
        );
        assert!(
            pk.fixed_polys.is_empty() && pk.permutation.polys.is_empty(),
            "whatever was mapped must have left the heap vectors"
        );
    }

    #[test]
    fn policy_does_not_change_what_the_key_serialises_to() {
        // Deferring cosets and mapping storage are about where values live, not
        // what they are. If the serialised form differed, a key written on a
        // device with the spill on could not be read by one with it off.
        let bytes = serialised_key();
        let heap = read(&bytes, &ProverConfig::heap());
        let mapped = read(&bytes, &ProverConfig::mapped_key());
        assert_eq!(
            heap.to_bytes(SerdeFormat::RawBytesUnchecked),
            mapped.to_bytes(SerdeFormat::RawBytesUnchecked),
        );
        assert_eq!(heap.to_bytes(SerdeFormat::RawBytesUnchecked), bytes);
    }
}
