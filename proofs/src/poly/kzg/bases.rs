//! Storage abstraction for the SRS bases (`g`, `g_lagrange`).
//!
//! `BasesStorage<C>` wraps either an owned `Vec<C>` (the default,
//! used by `unsafe_setup` and the eager `read_custom` path) or a
//! slice view into a memory-mapped file. Both variants `Deref` to
//! `&[C]` so existing MSM call sites in `poly/kzg/mod.rs` need no
//! change.
//!
//! The mmap variant is used by `ParamsKZG::read_mmap_arc`. It
//! reduces the heap footprint of the SRS to ~zero — at k=20 (BLS
//! 12-381, ~144 B per `G1Projective`) we save ~144 MiB per
//! materialised basis. The file mapping itself still counts toward
//! RSS when its pages are *touched*, but the OS evicts cold pages
//! under memory pressure (during heavy FFT phases the SRS pages
//! aren't referenced and get reclaimed). That OS-paging dance is
//! the difference between a 5 GiB peak (process killed on Android)
//! and a 4 GiB peak (process survives) at k=20-22.
//!
//! ## Layout invariant
//!
//! `BasesStorage::Mapped` is sound only when constructed against a
//! file whose bytes are a verbatim contiguous sequence of `C`s in
//! the same memory layout the producer used. We control that
//! format via
//! `ParamsKZG::write_mmap_companion` (not linked: it exists only under the
//! `mmap` feature, so a default-feature doc build cannot resolve it);
//! external callers must not feed arbitrary files to
//! `read_mmap_arc`.
//!
//! On BLS12-381 / midnight-curves the `G1Projective` /
//! `G1Affine` are both `#[repr(transparent)]` wrappers over a
//! C-bindings struct from `blst`, so the layout is platform-stable
//! and the cast is sound by construction.
//!
//! ## `unsafe` budget
//!
//! The crate-level `#![deny(unsafe_code)]` is opted out of for
//! this module — the mmap-as-slice cast and the manual
//! `Send`/`Sync` impls are the irreducible unsafe primitive that
//! the whole optimisation rests on. All `unsafe` blocks are
//! confined here behind invariants documented at construction.

// The crate denies `unsafe_code`; this module opts out **per site** rather
// than with a module-level `#![allow]`.
//
// The difference matters. A module-level opt-out makes every future `unsafe`
// in this file compile silently, which is precisely the warning
// `deny(unsafe_code)` exists to raise. Per-site, adding one is a deliberate
// act that shows up in a diff as its own line.
//
// Four sites, each with its own SAFETY note below.

use std::ops::Deref;
#[cfg(feature = "mmap")]
use std::sync::Arc;

#[cfg(feature = "mmap")]
use memmap2::Mmap;

/// Types this crate is willing to read and write as plain bytes.
///
/// The mmap companion format reinterprets file bytes as `C` and reinterprets
/// `&[C]` as bytes. Both directions require more than size and alignment:
///
/// - **Every bit pattern must be a valid `C`.** A type with a niche, an enum
///   discriminant, a `NonZero`, or any validity invariant makes the read
///   direction undefined behaviour for a file it did not produce — and the
///   header check cannot detect it, because the bytes are structurally fine.
/// - **`C` must have no padding.** Padding bytes are uninitialised; handing a
///   slice containing them to `write_all` is undefined behaviour and can leak
///   adjacent memory.
///
/// `ProcessedSerdeObject` guarantees neither — it is about serialisation, not
/// layout — so bounding on it was not enough. This trait is **sealed**, so only
/// this crate can add implementors, and it does so only for types whose layout
/// it has checked.
///
/// This is the same contract `bytemuck::Pod` expresses. It is spelled out here
/// rather than taking the dependency, and because the sealing is the point: a
/// downstream `Engine` must not be able to opt its own curve in.
#[cfg(feature = "mmap")]
pub trait PlainBytes: sealed::Sealed + Copy + 'static {}

#[cfg(feature = "mmap")]
mod sealed {
    /// Prevents implementation outside this crate.
    pub trait Sealed {}
}

// BLS12-381 G1 in projective form: three field elements, each a `[u64; 6]`
// wrapper with no padding and no niche, so every bit pattern is inhabited.
#[cfg(feature = "mmap")]
impl sealed::Sealed for midnight_curves::G1Projective {}
#[cfg(feature = "mmap")]
impl PlainBytes for midnight_curves::G1Projective {}

/// Storage backing for an SRS basis vector.
pub(crate) enum BasesStorage<C: 'static> {
    /// Heap-allocated. The default for `unsafe_setup`, the eager
    /// `read_custom`, and any `g_to_lagrange` recompute.
    Owned(Vec<C>),
    #[cfg(feature = "mmap")]
    /// Slice view into a memory-mapped file. The `Arc<Mmap>` keeps
    /// the mapping alive for the lifetime of this value.
    Mapped {
        /// Holds the mapping open. Cloning is cheap (`Arc::clone`)
        /// and increments the refcount; the mapping is dropped
        /// when the last clone falls out of scope.
        _mmap: Arc<Mmap>,
        /// Pointer into the mapping. Must be `align_of::<C>()`-
        /// aligned and remain valid for `len` consecutive `C`s.
        ptr: *const C,
        /// Number of `C`s reachable from `ptr`.
        len: usize,
    },
}

// SAFETY: the raw pointer in `Mapped` is sound to share across
// threads as long as `C: Send + Sync`. The mapping itself is
// Send + Sync (memmap2 doc); the pointer is immutable for the
// lifetime of the value.
#[allow(unsafe_code)]
unsafe impl<C: Send + Sync + 'static> Send for BasesStorage<C> {}
#[allow(unsafe_code)]
unsafe impl<C: Send + Sync + 'static> Sync for BasesStorage<C> {}

impl<C: 'static> BasesStorage<C> {
    /// Construct from an owned Vec — the common case.
    pub(crate) fn owned(v: Vec<C>) -> Self {
        BasesStorage::Owned(v)
    }

    /// Construct from a slice view into a memory-mapped file.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    ///
    /// 1. `ptr` points into the `mmap` region.
    /// 2. `ptr` has at least `align_of::<C>()` alignment.
    /// 3. The `len * size_of::<C>()` bytes starting at `ptr` are a valid
    ///    bit-pattern for `[C; len]` — i.e. the file was produced by
    ///    serialising live `C` values via the same in-memory representation
    ///    we're now claiming.
    /// 4. The `Arc<Mmap>` lives at least as long as any borrow obtained through
    ///    `Deref`.
    #[cfg(feature = "mmap")]
    #[allow(unsafe_code)]
    pub(crate) unsafe fn mapped(mmap: Arc<Mmap>, ptr: *const C, len: usize) -> Self {
        BasesStorage::Mapped {
            _mmap: mmap,
            ptr,
            len,
        }
    }

    /// `true` when the storage is backed by a memory-mapped file.
    /// Used by `downsize` to refuse in-place mutation.
    // Not dead: exercised by this module's tests. Clippy's dead-code pass on
    // the lib target alone does not see test usage, which is why the previous
    // `allow` looked justified and was not.
    #[cfg(test)]
    pub(crate) fn is_mapped(&self) -> bool {
        #[cfg(feature = "mmap")]
        {
            matches!(self, BasesStorage::Mapped { .. })
        }
        // Without the `mmap` feature there is no mapped variant to be.
        #[cfg(not(feature = "mmap"))]
        {
            false
        }
    }

    /// Truncate to `n` elements, allocating-copying out of an mmap
    /// region if necessary. `Vec::truncate` is in-place; for the
    /// mmap variant we must materialise into an owned Vec because
    /// the file mapping is read-only.
    pub(crate) fn truncate_into_owned(&mut self, n: usize)
    where
        C: Clone,
    {
        match self {
            BasesStorage::Owned(v) => v.truncate(n),
            #[cfg(feature = "mmap")]
            BasesStorage::Mapped { .. } => {
                let copy: Vec<C> = self.deref()[..n].to_vec();
                *self = BasesStorage::Owned(copy);
            }
        }
    }
}

impl<C: 'static> Deref for BasesStorage<C> {
    type Target = [C];
    fn deref(&self) -> &[C] {
        match self {
            BasesStorage::Owned(v) => v.as_slice(),
            // SAFETY: documented invariants in `mapped()` plus
            // `_mmap` keeps the region alive for our lifetime.
            #[cfg(feature = "mmap")]
            #[allow(unsafe_code)]
            BasesStorage::Mapped { ptr, len, .. } => unsafe {
                std::slice::from_raw_parts(*ptr, *len)
            },
        }
    }
}

impl<C: Clone + 'static> Clone for BasesStorage<C> {
    fn clone(&self) -> Self {
        match self {
            BasesStorage::Owned(v) => BasesStorage::Owned(v.clone()),
            // Cloning a Mapped is a refcount bump on the Arc plus
            // pointer copy — same backing region, no allocation.
            #[cfg(feature = "mmap")]
            BasesStorage::Mapped { _mmap, ptr, len } => BasesStorage::Mapped {
                _mmap: Arc::clone(_mmap),
                ptr: *ptr,
                len: *len,
            },
        }
    }
}

impl<C: std::fmt::Debug + 'static> std::fmt::Debug for BasesStorage<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self {
            BasesStorage::Owned(_) => "Owned",
            #[cfg(feature = "mmap")]
            BasesStorage::Mapped { .. } => "Mapped",
        };
        f.debug_struct("BasesStorage")
            .field("kind", &kind)
            .field("len", &self.deref().len())
            .finish()
    }
}

impl<C: PartialEq + 'static> PartialEq for BasesStorage<C> {
    fn eq(&self, other: &Self) -> bool {
        // Compare via slice equality — backing storage doesn't
        // matter for value equality.
        self.deref() == other.deref()
    }
}

impl<C: Eq + 'static> Eq for BasesStorage<C> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_round_trips() {
        let bs: BasesStorage<u32> = BasesStorage::owned(vec![1, 2, 3, 4]);
        assert_eq!(bs.len(), 4);
        assert_eq!(&*bs, &[1, 2, 3, 4]);
        assert!(!bs.is_mapped());
    }

    #[test]
    fn clone_preserves_data() {
        let bs: BasesStorage<u32> = BasesStorage::owned(vec![10, 20, 30]);
        let cloned = bs.clone();
        assert_eq!(&*bs, &*cloned);
    }

    #[test]
    fn truncate_into_owned_shortens() {
        let mut bs: BasesStorage<u32> = BasesStorage::owned(vec![1, 2, 3, 4, 5]);
        bs.truncate_into_owned(3);
        assert_eq!(&*bs, &[1, 2, 3]);
    }

    #[test]
    fn equality_uses_slice() {
        let a: BasesStorage<u32> = BasesStorage::owned(vec![1, 2, 3]);
        let b: BasesStorage<u32> = BasesStorage::owned(vec![1, 2, 3]);
        let c: BasesStorage<u32> = BasesStorage::owned(vec![1, 2, 4]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
