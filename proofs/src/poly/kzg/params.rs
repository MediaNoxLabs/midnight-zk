use std::{fmt::Debug, io, sync::OnceLock};

use ff::{Field, PrimeField};
use group::{prime::PrimeCurveAffine, Curve, Group};
use midnight_curves::pairing::{Engine, MultiMillerLoop};
use rand_core::RngCore;

use crate::{
    poly::{commitment::Params, kzg::bases::BasesStorage},
    utils::{
        arithmetic::{g_to_lagrange, parallelize},
        helpers::ProcessedSerdeObject,
        SerdeFormat,
    },
};

/// These are the public parameters for the polynomial commitment scheme.
///
/// `g_lagrange` is the inverse-FFT of `g`. It is a *derived* value, not
/// independent SRS material — anything we know about `g` determines
/// `g_lagrange` exactly. To keep the resident SRS as small as possible on
/// memory-constrained targets (mobile, wasm), `g_lagrange` is wrapped in
/// `OnceLock`: callers can release it via
/// [`drop_lazy_bases`](Self::drop_lazy_bases) between proofs and the next
/// `commit_lagrange` (or other Lagrange-basis access via the crate-private
/// `g_lagrange_slice`) will rebuild it from `g`.
///
/// File-format compatibility is preserved: `read_custom` still consumes
/// the `g_lagrange` bytes the writer produced and pre-populates the lock,
/// so callers see no behaviour change unless they opt in to dropping. We
/// document the byte layout so a future `read_custom_lazy` can `Seek` past
/// the Lagrange block to avoid even paying the load-time cost.
#[derive(Debug, Clone)]
pub struct ParamsKZG<E: Engine> {
    /// SRS monomial basis. Heap-allocated `Vec` in the eager path
    /// (`unsafe_setup`, `read_custom`); slice view into a memory-
    /// mapped file when constructed via `read_mmap_arc`.
    ///
    /// Not linked, because that method exists only under the `mmap` feature
    /// and a default-feature doc build would fail to resolve it.
    pub(crate) g: BasesStorage<E::G1>,
    /// SRS Lagrange basis. Lazy — see crate-level docs.
    /// `g_to_lagrange` only produces owned Vecs, so any
    /// lazy-recompute path always lands here as `Owned`.
    pub(crate) g_lagrange: OnceLock<BasesStorage<E::G1>>,
    pub(crate) g2: E::G2,
    pub(crate) s_g2: E::G2,
}

impl<E: Engine> Params for ParamsKZG<E>
where
    E::G1: Curve,
{
    fn max_k(&self) -> u32 {
        // `g_lagrange` may not yet be initialised — use `g` as the source
        // of truth for the SRS size. (When initialised, `g_lagrange.len()`
        // always equals `g.len()`.)
        self.g.len().ilog2()
    }

    fn downsize(&mut self, new_k: u32) {
        ParamsKZG::<E>::downsize(self, new_k)
    }
}

impl<E: Engine + Debug> ParamsKZG<E>
where
    E::G1: Curve,
{
    /// Returns the Lagrange-basis SRS as a slice. The first call after
    /// construction or after [`drop_lazy_bases`](Self::drop_lazy_bases)
    /// runs `g_to_lagrange(&g, log2(g.len()))` and caches the result.
    ///
    /// Internal hot-path accessor — `commit_lagrange` calls this rather
    /// than touching the field directly so the lazy-init contract is
    /// honoured uniformly.
    pub(crate) fn g_lagrange_slice(&self) -> &[E::G1] {
        // `&self.g` derefs to `&[E::G1]`; `g_to_lagrange` accepts a
        // slice. Wrap the produced `Vec` back into `BasesStorage`
        // so the lock's value type stays uniform.
        self.g_lagrange.get_or_init(|| {
            let k = self.g.len().ilog2();
            BasesStorage::owned(g_to_lagrange(&self.g, k))
        })
    }

    /// Release the cached Lagrange-basis SRS so the prover footprint
    /// drops back to just the monomial basis `g`. The next call to
    /// `commit_lagrange` (or any other `g_lagrange_slice` consumer) will
    /// recompute via FFT — pay CPU to win RAM.
    ///
    /// At BLS12-381 / projective storage the per-element cost is 144 B,
    /// so at k=20 this releases roughly 144 MiB. Recompute cost is one
    /// inverse-NTT over a 2^k domain (≈ tens-of-ms range on M-class
    /// silicon at k=14, scaling near-linearly).
    pub fn drop_lazy_bases(&mut self) {
        self.g_lagrange = OnceLock::new();
    }

    /// Downsize the current parameters to match a smaller `k`.
    pub fn downsize(&mut self, new_k: u32) {
        if self.max_k() == new_k {
            return;
        }

        let n = 1 << new_k;
        assert!((n as u64) < (1u64 << self.max_k()));
        // For mmap-backed `g` this materialises a fresh owned Vec
        // (the mapping is read-only). For owned `g` it's a normal
        // in-place truncate.
        self.g.truncate_into_owned(n);
        // Cached Lagrange basis is now stale (it was sized for the old
        // `g`). Reset; next consumer recomputes against the truncated
        // monomial basis.
        self.g_lagrange = OnceLock::new();
    }

    /// Initializes parameters for the curve, draws toxic secret from given rng.
    /// MUST NOT be used in production.
    pub fn unsafe_setup<R: RngCore>(k: u32, rng: R) -> Self {
        // Largest root of unity exponent of the Engine is `2^E::Fr::S`, so we can
        // only support FFTs of polynomials below degree `2^E::Fr::S`.
        assert!(k <= E::Fr::S);
        let n: u64 = 1 << k;

        // Calculate g = [G1, [s] G1, [s^2] G1, ..., [s^(n-1)] G1] in parallel.
        let g1 = E::G1::generator();
        let s = <E::Fr>::random(rng);

        let mut g = vec![E::G1::identity(); n as usize];
        parallelize(&mut g, |g, start| {
            let mut current_g: E::G1 = g1;
            current_g *= s.pow_vartime([start as u64]);
            for g in g.iter_mut() {
                *g = current_g;
                current_g *= s;
            }
        });

        let mut g_lagrange = vec![E::G1::identity(); n as usize];
        let mut root = E::Fr::ROOT_OF_UNITY;
        for _ in k..E::Fr::S {
            root = root.square();
        }
        let n_inv = E::Fr::from(n).invert().expect("inversion should be ok for n = 1<<k");
        let multiplier = (s.pow_vartime([n]) - E::Fr::ONE) * n_inv;
        parallelize(&mut g_lagrange, |g, start| {
            for (idx, g) in g.iter_mut().enumerate() {
                let offset = start + idx;
                let root_pow = root.pow_vartime([offset as u64]);
                let scalar = multiplier * root_pow * (s - root_pow).invert().unwrap();
                *g = g1 * scalar;
            }
        });

        let g2 = E::G2::generator();
        let s_g2 = g2 * s;

        Self {
            g: BasesStorage::owned(g),
            // Eagerly cache the Lagrange basis here — `unsafe_setup`
            // computed it anyway (line above), and skipping the cache
            // would just force recompute on first use with no win.
            g_lagrange: OnceLock::from(BasesStorage::owned(g_lagrange)),
            g2,
            s_g2,
        }
    }

    /// Initializes parameters for the curve through existing parameters
    /// k, g, g_lagrange (optional), g2, s_g2.
    ///
    /// When `g_lagrange` is `None` the Lagrange basis is deferred — it
    /// will be computed on first access. Pass `Some` only if you already
    /// have the values and want to skip the load-time FFT.
    pub fn from_parts(
        k: u32,
        g: Vec<E::G1>,
        g_lagrange: Option<Vec<E::G1>>,
        g2: E::G2,
        s_g2: E::G2,
    ) -> Self {
        let _ = k; // historically used to drive the FFT; now derived from g.len()
        Self {
            g: BasesStorage::owned(g),
            g_lagrange: match g_lagrange {
                Some(g_l) => OnceLock::from(BasesStorage::owned(g_l)),
                None => OnceLock::new(),
            },
            g2,
            s_g2,
        }
    }

    /// Returns the committed lagrange polynomials of these KZG params.
    ///
    /// Lazily materialises the basis if it has not been computed yet
    /// (or was released via [`drop_lazy_bases`](Self::drop_lazy_bases)).
    pub fn g_lagrange(&self) -> &[E::G1] {
        self.g_lagrange_slice()
    }

    /// Returns generator on G2
    pub fn g2(&self) -> E::G2 {
        self.g2
    }

    /// Returns \[τ\]₂, a commitment to τ in G2.
    pub fn s_g2(&self) -> E::G2 {
        self.s_g2
    }

    /// Writes parameters to buffer.
    ///
    /// Forces the Lagrange basis to be materialised before writing so the
    /// on-disk layout `[k, g[..], g_lagrange[..], g2, s_g2]` stays
    /// byte-for-byte compatible with v0.7.0 readers. Callers who want to
    /// elide the Lagrange block from disk should use a separate writer
    /// (none provided yet).
    pub fn write_custom<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()>
    where
        E::G1: Curve + ProcessedSerdeObject,
        E::G2: Curve + ProcessedSerdeObject,
    {
        writer.write_all(&self.g.len().ilog2().to_le_bytes())?;
        for el in self.g.iter() {
            el.write(writer, format)?;
        }
        for el in self.g_lagrange_slice().iter() {
            el.write(writer, format)?;
        }
        self.g2.write(writer, format)?;
        self.s_g2.write(writer, format)?;
        Ok(())
    }

    /// Reads params from a buffer.
    pub fn read_custom<R: io::Read>(reader: &mut R, format: SerdeFormat) -> io::Result<Self>
    where
        E::G1: Curve + ProcessedSerdeObject,
        E::G2: Curve + ProcessedSerdeObject,
    {
        let mut k = [0u8; 4];
        reader.read_exact(&mut k[..])?;
        let k = u32::from_le_bytes(k);
        let n = 1 << k;

        let (g, g_lagrange) = match format {
            SerdeFormat::Processed => {
                use group::GroupEncoding;
                let load_points_from_file_parallelly =
                    |reader: &mut R| -> io::Result<Vec<Option<E::G1>>> {
                        let mut points_compressed =
                            vec![<<E as Engine>::G1 as GroupEncoding>::Repr::default(); n];
                        for points_compressed in points_compressed.iter_mut() {
                            reader.read_exact((*points_compressed).as_mut())?;
                        }

                        let mut points = vec![Option::<E::G1>::None; n];
                        parallelize(&mut points, |points, chunks| {
                            for (i, point) in points.iter_mut().enumerate() {
                                *point =
                                    Option::from(E::G1::from_bytes(&points_compressed[chunks + i]));
                            }
                        });
                        Ok(points)
                    };

                let g = load_points_from_file_parallelly(reader)?;
                let g: Vec<<E as Engine>::G1> = g
                    .iter()
                    .map(|point| point.ok_or_else(|| io::Error::other("invalid point encoding")))
                    .collect::<Result<_, _>>()?;
                let g_lagrange = load_points_from_file_parallelly(reader)?;
                let g_lagrange: Vec<<E as Engine>::G1> = g_lagrange
                    .iter()
                    .map(|point| point.ok_or_else(|| io::Error::other("invalid point encoding")))
                    .collect::<Result<_, _>>()?;
                (g, g_lagrange)
            }
            SerdeFormat::RawBytes => {
                let g = (0..n)
                    .map(|_| <E::G1 as ProcessedSerdeObject>::read(reader, format))
                    .collect::<Result<Vec<_>, _>>()?;
                let g_lagrange = (0..n)
                    .map(|_| <E::G1 as ProcessedSerdeObject>::read(reader, format))
                    .collect::<Result<Vec<_>, _>>()?;
                (g, g_lagrange)
            }
            SerdeFormat::RawBytesUnchecked => {
                // avoid try branching for performance
                let g = (0..n)
                    .map(|_| <E::G1 as ProcessedSerdeObject>::read(reader, format).unwrap())
                    .collect::<Vec<_>>();
                let g_lagrange = (0..n)
                    .map(|_| <E::G1 as ProcessedSerdeObject>::read(reader, format).unwrap())
                    .collect::<Vec<_>>();
                (g, g_lagrange)
            }
        };

        let g2 = E::G2::read(reader, format)?;
        let s_g2 = E::G2::read(reader, format)?;

        // We just paid the I/O + parse cost to materialise `g_lagrange`
        // from disk — initialise the lock with it so the first commit
        // doesn't redundantly recompute. Callers who want to release
        // the basis between proofs invoke `drop_lazy_bases`.
        Ok(Self {
            g: BasesStorage::owned(g),
            g_lagrange: OnceLock::from(BasesStorage::owned(g_lagrange)),
            g2,
            s_g2,
        })
    }

    /// Reads params from a seekable buffer, **skipping the on-disk
    /// `g_lagrange` block entirely**. The Lagrange basis is left
    /// uninitialised; the first `commit_lagrange` (or
    /// `g_lagrange_slice` consumer) will recompute it via inverse-NTT
    /// of `g`.
    ///
    /// Memory profile vs. `read_custom`:
    ///
    /// - `read_custom` allocates two `Vec<E::G1>` of size 2^k during parsing.
    ///   Peak resident is 2× the SRS during load.
    /// - `read_custom_lazy` allocates only the `g` vector. Peak resident stays
    ///   at 1× the SRS during load.
    /// - The `Vec::drop` + recompute approach has the same final footprint as
    ///   `read_custom_lazy` *but* a 2× peak during load, plus the allocator
    ///   pool tends to keep the freed pages. Doing the skip at read time avoids
    ///   both costs.
    ///
    /// At k=20 (BLS12-381, 96 B per stored G1 element) this saves
    /// ~96 MiB peak; at k=22 it saves ~384 MiB. Critical on mobile
    /// where the OS kills the process well before peak heap maps to
    /// "total RAM".
    ///
    /// The block size is derived from `g`'s on-disk footprint — no
    /// per-element-size constants need to be hard-coded — so the
    /// method works for all three `SerdeFormat` variants.
    ///
    /// `R: Read + Seek`. `MidnightDataProvider`'s `BufReader<File>`
    /// satisfies both; in-memory `&[u8]` does via `io::Cursor<&[u8]>`.
    pub fn read_custom_lazy<R: io::Read + io::Seek>(
        reader: &mut R,
        format: SerdeFormat,
    ) -> io::Result<Self>
    where
        E::G1: Curve + ProcessedSerdeObject,
        E::G2: Curve + ProcessedSerdeObject,
    {
        let mut k = [0u8; 4];
        reader.read_exact(&mut k[..])?;
        let k = u32::from_le_bytes(k);
        let n = 1 << k;

        // Position before reading the `g` block. Used to compute the
        // matching block size for the seek-past-g_lagrange step below.
        let pos_before_g = reader.stream_position()?;

        let g: Vec<E::G1> = match format {
            SerdeFormat::Processed => {
                use group::GroupEncoding;
                let mut points_compressed =
                    vec![<<E as Engine>::G1 as GroupEncoding>::Repr::default(); n];
                for points_compressed in points_compressed.iter_mut() {
                    reader.read_exact((*points_compressed).as_mut())?;
                }
                let mut points = vec![Option::<E::G1>::None; n];
                parallelize(&mut points, |points, chunks| {
                    for (i, point) in points.iter_mut().enumerate() {
                        *point = Option::from(E::G1::from_bytes(&points_compressed[chunks + i]));
                    }
                });
                points
                    .into_iter()
                    .map(|point| point.ok_or_else(|| io::Error::other("invalid point encoding")))
                    .collect::<Result<_, _>>()?
            }
            SerdeFormat::RawBytes => (0..n)
                .map(|_| <E::G1 as ProcessedSerdeObject>::read(reader, format))
                .collect::<Result<Vec<_>, _>>()?,
            SerdeFormat::RawBytesUnchecked => (0..n)
                .map(|_| <E::G1 as ProcessedSerdeObject>::read(reader, format).unwrap())
                .collect::<Vec<_>>(),
        };

        // Seek past `g_lagrange` (same shape and size as the `g`
        // block we just consumed). No allocation, no parse, no
        // page-cache touch beyond what the disk subsystem reads
        // anyway for the file metadata.
        let pos_after_g = reader.stream_position()?;
        let block_size = pos_after_g - pos_before_g;
        reader.seek(io::SeekFrom::Current(block_size as i64))?;

        let g2 = E::G2::read(reader, format)?;
        let s_g2 = E::G2::read(reader, format)?;

        Ok(Self {
            g: BasesStorage::owned(g),
            // Empty lock — `g_lagrange_slice()` will populate via
            // inverse-NTT of `g` on first access.
            g_lagrange: OnceLock::new(),
            g2,
            s_g2,
        })
    }

    /// The identity a companion file must carry to be accepted by this build.
    ///
    /// # The rule, and what it does and does not detect
    ///
    /// The identity is a 64-bit FNV-1a over `(format version,
    /// type_name::<E::G1>, size_of::<E::G1>, align_of::<E::G1>, k)`. A file
    /// is accepted only if the value it recorded equals the value this
    /// build computes.
    ///
    /// **Detects:** a companion built for a different curve, a different point
    /// layout, a different format version, or a different `k`. Together with
    /// the `g_count == 2^k` check, that covers stale and incompatible files
    /// that are *structurally* plausible — the class `point_size` alone let
    /// through.
    ///
    /// **Does not detect:** a file whose points are the wrong *values* — an SRS
    /// from a different setup with the same curve and `k`. Detecting that needs
    /// a digest of the contents, and computing one at load means touching every
    /// page, which is precisely the cost the mapping exists to avoid. So the
    /// rule is deliberately *identity, not integrity*: the consumer is
    /// responsible for keying the file by the SRS it was derived from, and the
    /// trust-boundary docs on `read_mmap_arc` say the file is a local cache
    /// this library produced, never something fetched.
    ///
    /// `type_name` is not guaranteed stable across compiler releases. That can
    /// only produce a **false rejection** — a cache miss and a rebuild — never
    /// a false accept, and a compiler change is exactly when the in-memory
    /// layout this format depends on is allowed to change (F-024), so refusing
    /// there is the conservative behaviour, not a defect.
    #[cfg(feature = "mmap")]
    pub(crate) fn layout_identity(k: u32) -> u64 {
        const VERSION: u32 = 2;
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut feed = |b: &[u8]| {
            for &x in b {
                h ^= x as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        feed(&VERSION.to_le_bytes());
        feed(std::any::type_name::<E::G1>().as_bytes());
        feed(&(std::mem::size_of::<E::G1>() as u64).to_le_bytes());
        feed(&(std::mem::align_of::<E::G1>() as u64).to_le_bytes());
        feed(&k.to_le_bytes());
        h
    }

    /// Reads params from a memory-mapped companion file produced by
    /// [`write_mmap_companion`](Self::write_mmap_companion).
    ///
    /// Constructs `ParamsKZG` with `g` and (if present) `g_lagrange`
    /// pointing into the mmap region — zero heap allocation for the
    /// SRS. The mmap is held alive via `Arc<Mmap>`; clones of the
    /// `ParamsKZG` share the same backing region.
    ///
    /// # Format
    ///
    /// 64-byte header (little-endian throughout):
    ///
    /// ```text
    ///  0..8   magic              = b"MDNGHTV2"
    ///  8..12  version: u32       = 1
    /// 12..16  k: u32
    /// 16..20  point_size: u32    = size_of::<E::G1>() — guard against layout drift
    /// 20..24  flags: u32         (bit 0 = has g_lagrange)
    /// 24..32  identity: u64      = layout_identity(k) — see that fn for the rule
    /// 32..40  g_offset: u64      = 64
    /// 40..48  g_count: u64       = 1 << k
    /// 48..56  g_lagrange_offset: u64
    /// 56..64  g_lagrange_count: u64
    /// 64..72  g2_offset: u64
    /// 72..80  g2_size:   u64
    /// 80..88  s_g2_offset: u64
    /// 88..96  s_g2_size: u64
    /// 96..g_offset   padding (`g_offset` defaults to 128 for SIMD alignment)
    /// ```
    ///
    /// Body: `g` and `g_lagrange` are raw bytes of consecutive `E::G1`
    /// values in the in-memory representation. `g2` / `s_g2` use the
    /// `ProcessedSerdeObject::Repr` encoding so the file can be
    /// produced and consumed without committing to a specific
    /// `SerdeFormat`.
    ///
    /// # Trust boundary
    ///
    /// **This file is a local cache this library produced. It is not an
    /// interchange format.** Do not fetch it over a network, sync it between
    /// devices, accept it from another party, or place it anywhere a less
    /// privileged process can write. Treat it exactly as you would treat the
    /// process's own heap.
    ///
    /// What the header check buys, precisely:
    ///
    /// - the magic, `point_size`, every offset and count, and the alignment of
    ///   each block are validated before anything is mapped, so a truncated,
    ///   mismatched or out-of-range file fails with `InvalidData` rather than
    ///   producing UB at MSM time;
    /// - the *contents* are not validated. Bytes inside a well-formed block are
    ///   reinterpreted as `E::G1` without checking that they encode points on
    ///   the curve.
    ///
    /// So a hostile file with a valid header yields **wrong proofs, not memory
    /// corruption** — and that holds only because every bit pattern is a valid
    /// inhabitant of `E::G1`'s field-element representation. A curve type with
    /// niches or validity invariants would make the same file a soundness
    /// problem, which is why `point_size` is checked and why this is stated
    /// rather than left implicit.
    ///
    /// The caller is also trusting the producer to have used the same `E::G1`
    /// memory layout; `point_size` in the header is what makes a mismatched
    /// producer and consumer fail fast.
    // Confined `unsafe`: mmap-region pointer arithmetic +
    // `from_raw_parts`-style slice construction. The invariants are
    // documented inline; see also `BasesStorage::mapped` SAFETY notes.
    #[allow(unsafe_code)]
    #[cfg(feature = "mmap")]
    pub fn read_mmap_arc(mmap: std::sync::Arc<memmap2::Mmap>) -> io::Result<Self>
    where
        E::G2: ProcessedSerdeObject,
        // Reinterpreting file bytes as `E::G1` is only defined when every bit
        // pattern inhabits the type. `ProcessedSerdeObject` does not say that —
        // it is about serialisation — so a curve with a niche would make a
        // structurally valid file undefined behaviour on first dereference.
        // `PlainBytes` is sealed, so only curves this crate has checked qualify.
        E::G1: crate::poly::kzg::bases::PlainBytes,
    {
        const HEADER_LEN: usize = 96;
        const MAGIC: &[u8; 8] = b"MDNGHTV2";
        const VERSION: u32 = 2;

        fn bad(msg: &'static str) -> io::Error {
            io::Error::new(io::ErrorKind::InvalidData, msg)
        }
        // `u64 -> usize` is a narrowing on 32-bit targets. `as` silently wraps —
        // `2^32 + 128` becomes `128` and validation then succeeds against the
        // wrong region — so every header value goes through `try_from`.
        fn field_usize(bytes: &[u8], at: usize, what: &'static str) -> io::Result<usize> {
            let raw = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
            usize::try_from(raw).map_err(|_| bad(what))
        }
        fn field_u32(bytes: &[u8], at: usize) -> u32 {
            u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
        }
        // A block is `[off, off + len)`; both the end and the containment are
        // checked, never saturated — saturation turns an overflowing length into
        // a plausible one.
        fn block<'a>(
            bytes: &'a [u8],
            off: usize,
            len: usize,
            what: &'static str,
        ) -> io::Result<&'a [u8]> {
            let end = off.checked_add(len).ok_or_else(|| bad(what))?;
            if end > bytes.len() {
                return Err(bad(what));
            }
            Ok(&bytes[off..end])
        }

        let bytes: &[u8] = &mmap[..];
        if bytes.len() < HEADER_LEN {
            return Err(bad("header truncated"));
        }
        if &bytes[0..8] != MAGIC {
            return Err(bad("bad magic — not a companion file, or a v1 companion, which this reader refuses rather than misreads"));
        }
        if field_u32(bytes, 8) != VERSION {
            return Err(bad("unsupported companion version"));
        }
        let k = field_u32(bytes, 12);
        let point_size = usize::try_from(field_u32(bytes, 16)).map_err(|_| bad("point_size"))?;
        if point_size != std::mem::size_of::<E::G1>() {
            return Err(bad(
                "point_size mismatch — companion file built for a different layout",
            ));
        }
        let flags = field_u32(bytes, 20);
        let has_g_lagrange = (flags & 0x1) != 0;
        // Identity (bytes 24..32): see `layout_identity` for the rule and its limits.
        let identity = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
        if identity != Self::layout_identity(k) {
            return Err(bad(
                "companion identity mismatch — built for a different curve, layout, or k",
            ));
        }

        let g_off = field_usize(bytes, 32, "g_offset")?;
        let g_count = field_usize(bytes, 40, "g_count")?;
        let gl_off = field_usize(bytes, 48, "g_lagrange_offset")?;
        let gl_count = field_usize(bytes, 56, "g_lagrange_count")?;
        let g2_off = field_usize(bytes, 64, "g2_offset")?;
        let g2_size = field_usize(bytes, 72, "g2_size")?;
        let s_g2_off = field_usize(bytes, 80, "s_g2_offset")?;
        let s_g2_size = field_usize(bytes, 88, "s_g2_size")?;

        // k / count consistency: the monomial basis has exactly 2^k points and
        // the Lagrange basis, if present, the same.
        let expected = 1usize.checked_shl(k).ok_or_else(|| bad("k too large"))?;
        if k >= usize::BITS || g_count != expected {
            return Err(bad("g_count does not match 2^k"));
        }
        if has_g_lagrange && gl_count != expected {
            return Err(bad("g_lagrange_count does not match 2^k"));
        }

        let g_len = g_count.checked_mul(point_size).ok_or_else(|| bad("g range overflow"))?;
        let g_block = block(bytes, g_off, g_len, "g block out of range")?;
        // Alignment is not implied by the layout: block starts depend on
        // `size_of::<E::G1>()` and need not land on `align_of` for every type.
        // Checked on the *address*, with a checked add.
        let g_addr = (bytes.as_ptr() as usize)
            .checked_add(g_off)
            .ok_or_else(|| bad("g address overflow"))?;
        if g_addr % std::mem::align_of::<E::G1>() != 0 {
            return Err(bad("g block misaligned for E::G1"));
        }
        let g_ptr = g_block.as_ptr() as *const E::G1;
        // SAFETY: `g_block` lies inside the mapping, is `g_count * size_of` bytes,
        // and starts on an `align_of::<E::G1>()` boundary (all checked above);
        // `E::G1: PlainBytes` asserts every bit pattern is a valid value with no
        // padding; the `Arc` clone keeps the mapping alive for the storage.
        let g = unsafe { BasesStorage::mapped(mmap.clone(), g_ptr, g_count) };

        let g_lagrange = if has_g_lagrange {
            let gl_len =
                gl_count.checked_mul(point_size).ok_or_else(|| bad("gl range overflow"))?;
            let gl_block = block(bytes, gl_off, gl_len, "g_lagrange block out of range")?;
            let gl_addr = (bytes.as_ptr() as usize)
                .checked_add(gl_off)
                .ok_or_else(|| bad("gl address overflow"))?;
            if gl_addr % std::mem::align_of::<E::G1>() != 0 {
                return Err(bad("g_lagrange block misaligned for E::G1"));
            }
            let gl_ptr = gl_block.as_ptr() as *const E::G1;
            // SAFETY: as for `g`.
            let gl = unsafe { BasesStorage::mapped(mmap.clone(), gl_ptr, gl_count) };
            Some(gl)
        } else {
            None
        };
        // The Lagrange basis is a `OnceLock` so `read_custom_lazy` can leave it
        // unset and recompute on first use; a companion that carries it is set
        // eagerly here, one that does not leaves it for the same lazy path.
        let g_lagrange = {
            let cell = std::sync::OnceLock::new();
            if let Some(gl) = g_lagrange {
                let _ = cell.set(gl);
            }
            cell
        };

        // g2 / s_g2 use the SerdeFormat encoding, not the raw layout, so they are
        // parsed rather than mapped — but their ranges are still checked first,
        // so a truncated file returns `InvalidData` instead of panicking on the
        // slice.
        let mut g2_slice = block(bytes, g2_off, g2_size, "g2 block out of range")?;
        let mut s_g2_slice = block(bytes, s_g2_off, s_g2_size, "s_g2 block out of range")?;
        let g2 = E::G2::read(&mut g2_slice, SerdeFormat::RawBytesUnchecked)?;
        let s_g2 = E::G2::read(&mut s_g2_slice, SerdeFormat::RawBytesUnchecked)?;

        Ok(Self {
            g,
            g_lagrange,
            g2,
            s_g2,
        })
    }

    /// Writes the in-memory `ParamsKZG` to a companion file laid out
    /// for [`read_mmap_arc`](Self::read_mmap_arc). Includes both `g` and
    /// `g_lagrange` (the latter is force-materialised via
    /// `g_lagrange_slice`).
    ///
    /// Disk cost at BLS12-381 / k=20 / projective storage:
    /// 64 B header + 2 × 2^20 × 144 B + g2 + s_g2 ≈ 288 MiB. Bigger
    /// than the original CDN file (which uses 96-byte affine on
    /// disk) because we mirror the in-memory layout for zero-copy
    /// loading. Disk is cheap on mobile (94 GB free on the S24);
    /// the memory win at proof time is the goal.
    // Confined `unsafe`: reinterprets the `[E::G1]` slice as `[u8]`
    // for raw bytewise write. Sound because `E::G1` is
    // `#[repr(transparent)]` over a C-layout struct (verified for
    // midnight-curves' `G1Projective`).
    #[allow(unsafe_code)]
    #[cfg(feature = "mmap")]
    pub fn write_mmap_companion<W: io::Write>(&self, writer: &mut W) -> io::Result<()>
    where
        E::G2: ProcessedSerdeObject,
        E::G1: ProcessedSerdeObject,
        // Reading `&[E::G1]` back as bytes is only defined when the type has no
        // padding: padding bytes are uninitialised, and handing them to
        // `write_all` is undefined behaviour that can leak adjacent memory.
        // Same sealed trait, same reason.
        E::G1: crate::poly::kzg::bases::PlainBytes,
    {
        const HEADER_LEN: usize = 96;
        const G_OFFSET: u64 = 128; // 128-byte alignment for SIMD friendliness

        let n = self.g.len() as u64;
        let k = self.g.len().ilog2();
        let point_size = std::mem::size_of::<E::G1>() as u64;
        let g_block_bytes = n * point_size;

        // Force g_lagrange materialisation so we can write it.
        let g_lagrange = self.g_lagrange_slice();
        let gl_count = g_lagrange.len() as u64;
        let gl_offset = G_OFFSET + g_block_bytes;
        let gl_block_bytes = gl_count * point_size;
        let g2_offset = gl_offset + gl_block_bytes;
        // Serialise g2/s_g2 into temp buffers so we know their sizes.
        let mut g2_buf = Vec::new();
        self.g2.write(&mut g2_buf, SerdeFormat::RawBytesUnchecked)?;
        let mut s_g2_buf = Vec::new();
        self.s_g2.write(&mut s_g2_buf, SerdeFormat::RawBytesUnchecked)?;
        let g2_size = g2_buf.len() as u64;
        let s_g2_offset = g2_offset + g2_size;
        let s_g2_size = s_g2_buf.len() as u64;

        // Header
        // Version 2: the eight bytes v1 reserved now carry the layout identity,
        // and the magic changed with it so a v1 file is *refused*, not read
        // with a zero identity that happens not to match.
        writer.write_all(b"MDNGHTV2")?;
        writer.write_all(&2u32.to_le_bytes())?; // version
        writer.write_all(&k.to_le_bytes())?;
        writer.write_all(&(point_size as u32).to_le_bytes())?;
        writer.write_all(&1u32.to_le_bytes())?; // flags: has_g_lagrange
        writer.write_all(&Self::layout_identity(k).to_le_bytes())?; // identity
        writer.write_all(&G_OFFSET.to_le_bytes())?;
        writer.write_all(&n.to_le_bytes())?;
        writer.write_all(&gl_offset.to_le_bytes())?;
        writer.write_all(&gl_count.to_le_bytes())?;
        writer.write_all(&g2_offset.to_le_bytes())?;
        writer.write_all(&g2_size.to_le_bytes())?;
        writer.write_all(&s_g2_offset.to_le_bytes())?;
        writer.write_all(&s_g2_size.to_le_bytes())?;
        // Pad to G_OFFSET
        let pad = G_OFFSET as usize - HEADER_LEN;
        writer.write_all(&vec![0u8; pad])?;

        // g block: raw bytes of consecutive E::G1 in memory layout
        let g_slice: &[E::G1] = &self.g;
        let g_bytes = unsafe {
            std::slice::from_raw_parts(
                g_slice.as_ptr() as *const u8,
                std::mem::size_of_val(g_slice),
            )
        };
        writer.write_all(g_bytes)?;

        // g_lagrange block
        let gl_bytes = unsafe {
            std::slice::from_raw_parts(
                g_lagrange.as_ptr() as *const u8,
                std::mem::size_of_val(g_lagrange),
            )
        };
        writer.write_all(gl_bytes)?;

        // g2 / s_g2 — write the serialised buffers we built above
        writer.write_all(&g2_buf)?;
        writer.write_all(&s_g2_buf)?;

        Ok(())
    }
}

// TODO: see the issue at https://github.com/appliedzkp/halo2/issues/45
// So we probably need much smaller verifier key. However for new bases in g1
// should be in verifier keys.
/// KZG multi-open verification parameters
#[derive(Clone, Debug)]
pub struct ParamsVerifierKZG<E: MultiMillerLoop> {
    pub(crate) s_g2: E::G2,
    pub(crate) n_g2_prepared: E::G2Prepared,
    pub(crate) s_g2_prepared: E::G2Prepared,
}

impl<E: MultiMillerLoop + Debug> ParamsVerifierKZG<E>
where
    E::G2: Curve + ProcessedSerdeObject,
{
    /// Returns \[τ\]₂, a commitment to τ in G2.
    pub fn s_g2(&self) -> E::G2 {
        self.s_g2
    }

    /// Writes parameters to buffer
    pub fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()> {
        self.s_g2.write(writer, format)?;
        Ok(())
    }

    /// Reads params from a buffer.
    pub fn read<R: io::Read>(reader: &mut R, format: SerdeFormat) -> io::Result<Self> {
        let s_g2 = E::G2::read(reader, format)?;
        let s_g2_prepared = E::G2Prepared::from(s_g2.into());
        let n_g2_prepared = E::G2Prepared::from(-E::G2Affine::generator());

        Ok(Self {
            s_g2,
            n_g2_prepared,
            s_g2_prepared,
        })
    }
}

impl<E: MultiMillerLoop + Debug> ParamsKZG<E> {
    /// Consume the prover parameters into verifier parameters. Need to specify
    /// the size of public inputs.
    pub fn verifier_params(&self) -> ParamsVerifierKZG<E> {
        let n_g2_prepared = E::G2Prepared::from((-self.g2).into());
        let s_g2_prepared = E::G2Prepared::from(self.s_g2.into());
        ParamsVerifierKZG {
            s_g2: self.s_g2,
            n_g2_prepared,
            s_g2_prepared,
        }
    }
}

#[cfg(test)]
mod test {
    use rand_core::OsRng;

    use crate::{
        poly::{
            commitment::PolynomialCommitmentScheme,
            kzg::{params::ParamsKZG, KZGCommitmentScheme},
        },
        utils::SerdeFormat,
    };

    #[test]
    fn test_commit_lagrange() {
        const K: u32 = 6;

        use midnight_curves::{Bls12, Fq};

        use crate::poly::EvaluationDomain;

        let params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(K, OsRng);
        let domain = EvaluationDomain::new(1, K);

        let mut a = domain.empty_lagrange();

        for (i, a) in a.iter_mut().enumerate() {
            *a = Fq::from(i as u64);
        }

        let b = domain.lagrange_to_coeff(a.clone());

        let tmp = KZGCommitmentScheme::commit_lagrange(&params, &a);
        let commitment = KZGCommitmentScheme::commit(&params, &b);

        assert_eq!(commitment, tmp);
    }

    #[test]
    fn test_parameter_serialisation_roundtrip() {
        const K: u32 = 4;

        use midnight_curves::Bls12;

        let params0: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(K, OsRng);
        let mut data = vec![];
        ParamsKZG::write_custom(&params0, &mut data, SerdeFormat::RawBytesUnchecked).unwrap();
        let params1 =
            ParamsKZG::<Bls12>::read_custom::<_>(&mut &data[..], SerdeFormat::RawBytesUnchecked)
                .unwrap();

        assert_eq!(params0.g.len(), params1.g.len());
        // Materialise the Lagrange basis on both sides before comparing —
        // the field is `OnceLock` now, so we go through the lazy accessor.
        assert_eq!(
            params0.g_lagrange_slice().len(),
            params1.g_lagrange_slice().len()
        );

        assert_eq!(params0.g, params1.g);
        assert_eq!(params0.g_lagrange_slice(), params1.g_lagrange_slice());
        assert_eq!(params0.g2, params1.g2);
        assert_eq!(params0.s_g2, params1.s_g2);
    }

    /// Verifies that `read_custom_lazy` reads back the same `g`, `g2`,
    /// `s_g2` as `read_custom`, and that the deferred Lagrange basis
    /// — computed via FFT on first access — equals the eagerly-read
    /// basis. Confirms both that the seek-past-g_lagrange step lands
    /// at the right offset and that the lazy recompute is correct.
    #[test]
    fn test_read_custom_lazy_matches_eager() {
        const K: u32 = 5;

        use midnight_curves::Bls12;

        let params_eager: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(K, OsRng);
        let mut buf = Vec::new();
        ParamsKZG::write_custom(&params_eager, &mut buf, SerdeFormat::RawBytesUnchecked).unwrap();

        let mut cursor = std::io::Cursor::new(&buf[..]);
        let params_lazy =
            ParamsKZG::<Bls12>::read_custom_lazy(&mut cursor, SerdeFormat::RawBytesUnchecked)
                .unwrap();

        // Monomial basis + G2 components match byte-for-byte.
        assert_eq!(params_eager.g, params_lazy.g);
        assert_eq!(params_eager.g2, params_lazy.g2);
        assert_eq!(params_eager.s_g2, params_lazy.s_g2);

        // Cursor must be at end-of-buffer — seek landed correctly.
        assert_eq!(cursor.position() as usize, buf.len());

        // Lagrange basis from FFT recompute equals the eager parse.
        assert_eq!(
            params_eager.g_lagrange_slice(),
            params_lazy.g_lagrange_slice()
        );
    }

    /// Verifies that the lazy Lagrange basis matches an eager
    /// computation, and that `drop_lazy_bases` + `g_lagrange_slice`
    /// round-trips back to the same values via FFT.
    #[test]
    fn test_drop_lazy_bases_round_trip() {
        const K: u32 = 5;

        use midnight_curves::Bls12;

        use crate::poly::EvaluationDomain;

        let mut params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(K, OsRng);
        let _ = EvaluationDomain::<midnight_curves::Fq>::new(1, K);

        // Force-materialise via the accessor.
        let eager: Vec<_> = params.g_lagrange_slice().to_vec();
        assert_eq!(eager.len(), 1usize << K);

        // Drop the cache, then recompute from `g` via FFT — must equal.
        params.drop_lazy_bases();
        let recomputed: &[midnight_curves::G1Projective] = params.g_lagrange_slice();
        assert_eq!(eager.as_slice(), recomputed);
    }

    /// Round-trip a `ParamsKZG` through the companion mmap format.
    /// Verifies that:
    ///  1. `write_mmap_companion` produces a file that `read_mmap_arc` accepts;
    ///  2. The reconstructed `g`, `g_lagrange`, `g2`, `s_g2` all equal the
    ///     originals byte-for-byte;
    ///  3. The reconstructed `g` is genuinely backed by the mmap (not silently
    ///     copied into an owned `Vec`).
    #[cfg(feature = "mmap")]
    #[test]
    fn test_mmap_companion_round_trip() {
        const K: u32 = 5;
        use std::{io::Write as _, sync::Arc};

        use midnight_curves::Bls12;

        let params0: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(K, OsRng);

        // Write to a tempfile so we can mmap it back.
        let mut tmp = tempfile::NamedTempFile::new().expect("temp file");
        params0.write_mmap_companion(&mut tmp).expect("write companion");
        tmp.flush().expect("flush");
        let file = tmp.reopen().expect("reopen");
        // SAFETY: we control the file lifecycle; mmap is read-only.
        #[allow(unsafe_code)]
        let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file) }.expect("mmap"));

        let params1 = ParamsKZG::<Bls12>::read_mmap_arc(mmap).expect("read mmap");

        // Equality through the BasesStorage Deref → slice compare.
        assert_eq!(params0.g.len(), params1.g.len());
        assert_eq!(&*params0.g, &*params1.g);
        assert_eq!(params0.g_lagrange_slice(), params1.g_lagrange_slice(),);
        assert_eq!(params0.g2, params1.g2);
        assert_eq!(params0.s_g2, params1.s_g2);

        // Confirm the storage really is mmap-backed (not silently
        // copied) — the whole point of the optimisation.
        use crate::poly::kzg::bases::BasesStorage;
        assert!(
            matches!(params1.g, BasesStorage::Mapped { .. }),
            "g should be mmap-backed after read_mmap_arc"
        );
    }
}

#[cfg(all(test, feature = "mmap"))]
mod companion_rejection {
    //! Every malformed companion must come back as `InvalidData`, never a
    //! panic. Built by taking a good file and damaging one thing at a time.

    use std::sync::Arc;

    use midnight_curves::Bls12;
    use rand_core::OsRng;

    use super::ParamsKZG;

    const K: u32 = 4;

    fn good() -> Vec<u8> {
        let params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(K, OsRng);
        let mut buf = Vec::new();
        params.write_mmap_companion(&mut buf).expect("write companion");
        buf
    }

    fn load(bytes: Vec<u8>) -> std::io::Result<ParamsKZG<Bls12>> {
        // An anonymous mapping stands in for the file; `read_mmap_arc` only
        // sees `&[u8]` plus the `Arc`, so this exercises the same code.
        let mut m = memmap2::MmapMut::map_anon(bytes.len().max(1)).expect("map_anon");
        m[..bytes.len()].copy_from_slice(&bytes);
        let ro = m.make_read_only().expect("make_read_only");
        ParamsKZG::<Bls12>::read_mmap_arc(Arc::new(ro))
    }

    fn assert_rejected(bytes: Vec<u8>, what: &str) {
        match load(bytes) {
            Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::InvalidData, "{what}: {e}"),
            Ok(_) => panic!("{what}: a damaged companion was accepted"),
        }
    }

    fn put_u64(b: &mut [u8], at: usize, v: u64) {
        b[at..at + 8].copy_from_slice(&v.to_le_bytes());
    }
    fn put_u32(b: &mut [u8], at: usize, v: u32) {
        b[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }

    #[test]
    fn the_undamaged_file_loads() {
        let params = load(good()).expect("good companion must load");
        assert_eq!(params.g.len(), 1 << K);
    }

    #[test]
    fn truncated_header_is_invalid_data() {
        let g = good();
        assert_rejected(g[..40].to_vec(), "header cut short");
        assert_rejected(Vec::new(), "empty file");
    }

    #[test]
    fn truncated_body_is_invalid_data_not_a_panic() {
        // The g2 / s_g2 blocks sit at the end. Cutting the file inside them used
        // to reach `&bytes[off..off + size]` unchecked and panic; now it must
        // come back as an error.
        let g = good();
        for cut in [g.len() - 1, g.len() - 48, g.len() / 2, 96] {
            assert_rejected(g[..cut].to_vec(), &format!("body truncated at {cut}"));
        }
    }

    #[test]
    fn v1_magic_is_refused_rather_than_misread() {
        let mut g = good();
        g[..8].copy_from_slice(b"MDNGHTV1");
        assert_rejected(g, "v1 magic");
    }

    #[test]
    fn wrong_version_is_rejected() {
        let mut g = good();
        put_u32(&mut g, 8, 1);
        assert_rejected(g, "version 1 with v2 magic");
    }

    #[test]
    fn wrong_identity_is_rejected() {
        let mut g = good();
        let id = u64::from_le_bytes(g[24..32].try_into().unwrap());
        put_u64(&mut g, 24, id ^ 1);
        assert_rejected(g, "identity off by one bit");
    }

    #[test]
    fn k_and_count_must_agree() {
        // Same identity as recorded for the *wrong* k would be needed to get
        // past the identity check, so damage the count instead: identity says
        // k=4, g_count says otherwise.
        let mut g = good();
        put_u64(&mut g, 40, (1u64 << K) + 1);
        assert_rejected(g, "g_count != 2^k");
    }

    #[test]
    fn range_overflow_is_an_error_not_a_wrap() {
        // A count that overflows `count * point_size` must not saturate into a
        // plausible length and pass the containment check.
        let mut g = good();
        put_u64(&mut g, 40, u64::MAX / 2);
        assert_rejected(g, "g_count * point_size overflows");
        let mut g = good();
        put_u64(&mut g, 72, u64::MAX);
        assert_rejected(g, "g2_size overflows the file");
        let mut g = good();
        put_u64(&mut g, 80, u64::MAX - 8);
        assert_rejected(g, "s_g2_offset near u64::MAX");
    }

    #[test]
    fn misaligned_g_block_is_rejected() {
        // Point the g block one byte off its aligned start. The block still
        // lies inside the file, so only the alignment check can catch it.
        let mut g = good();
        let g_off = u64::from_le_bytes(g[32..40].try_into().unwrap());
        put_u64(&mut g, 32, g_off + 1);
        assert_rejected(g, "g block misaligned by one byte");
    }

    #[test]
    fn identity_pins_k() {
        // Two setups at different k produce different identities, so a k=4
        // file cannot be passed off as k=5 by editing the header's k alone.
        assert_ne!(
            ParamsKZG::<Bls12>::layout_identity(4),
            ParamsKZG::<Bls12>::layout_identity(5)
        );
    }
}
