//! Read-only polynomial access shared by owned and mapped storage.

use std::{fmt, marker::PhantomData, ops::Deref};

use ff::Field;

use super::Polynomial;

/// Minimal interface required by proving paths that only read polynomial
/// values. Implementations may own their values or expose a checked view over
/// another backing store.
pub(crate) trait PolynomialRead<F> {
    /// Representation/basis carried through every borrowed view.
    type Basis;

    fn values(&self) -> &[F];
}

impl<F, B> PolynomialRead<F> for Polynomial<F, B> {
    type Basis = B;

    fn values(&self) -> &[F] {
        &self.values
    }
}

/// A copyable, basis-typed read-only polynomial reference. Hot loops consume
/// this concrete type, so choosing owned versus mapped storage happens once at
/// the boundary and adds no virtual dispatch per field operation.
#[derive(Clone, Copy)]
pub(crate) struct PolynomialView<'a, F, B> {
    values: &'a [F],
    _marker: PhantomData<B>,
}

impl<'a, F, B> PolynomialView<'a, F, B> {
    pub(crate) fn new(values: &'a [F]) -> Self {
        Self {
            values,
            _marker: PhantomData,
        }
    }

    /// Materialise one owned polynomial. Spill callers use this only for the
    /// single polynomial currently being transformed, preserving the bounded
    /// peak-memory property.
    pub(crate) fn materialise(self) -> Polynomial<F, B>
    where
        F: Clone,
    {
        Polynomial {
            values: self.values.to_vec(),
            _marker: PhantomData,
        }
    }
}

impl<F, B> Deref for PolynomialView<'_, F, B> {
    type Target = [F];

    fn deref(&self) -> &[F] {
        self.values
    }
}

impl<F: fmt::Debug, B> fmt::Debug for PolynomialView<'_, F, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.values.fmt(f)
    }
}

impl<F, B> PolynomialRead<F> for PolynomialView<'_, F, B> {
    type Basis = B;

    fn values(&self) -> &[F] {
        self.values
    }
}

/// Adapt any homogeneous batch to the one concrete representation used by
/// read-only proving code. The allocation contains only slice fat-pointers;
/// coefficient storage is never copied.
pub(crate) fn polynomial_views<F, P>(polys: &[P]) -> Vec<PolynomialView<'_, F, P::Basis>>
where
    P: PolynomialRead<F>,
{
    polys.iter().map(|poly| PolynomialView::new(poly.values())).collect()
}

/// Compute a linear combination of read-only polynomials into one owned result.
///
/// This is the mapped-storage counterpart of the generic `inner_product`
/// helper. It allocates exactly the result polynomial instead of materialising
/// every mapped input before combining them.
pub(crate) fn polynomial_inner_product<F, B>(
    polys: &[PolynomialView<'_, F, B>],
    mut scalars: impl Iterator<Item = F>,
) -> Polynomial<F, B>
where
    F: Field,
{
    let first = polys.first().expect("polynomial inner product cannot be empty");
    let first_scalar = scalars.next().expect("polynomial inner product needs a scalar");
    let mut values: Vec<F> = first.iter().map(|value| *value * first_scalar).collect();

    for (poly, scalar) in polys[1..].iter().zip(scalars) {
        assert_eq!(poly.len(), values.len(), "polynomial lengths must agree");
        for (acc, value) in values.iter_mut().zip(poly.iter()) {
            *acc += *value * scalar;
        }
    }

    Polynomial {
        values,
        _marker: PhantomData,
    }
}

#[cfg(test)]
mod tests {
    use midnight_curves::Fq;

    use super::*;
    use crate::{poly::Coeff, utils::arithmetic::inner_product};

    #[test]
    fn view_inner_product_matches_owned_polynomials() {
        let polys = vec![
            Polynomial::<Fq, Coeff> {
                values: vec![Fq::from(1), Fq::from(2), Fq::from(3)],
                _marker: PhantomData,
            },
            Polynomial::<Fq, Coeff> {
                values: vec![Fq::from(4), Fq::from(5), Fq::from(6)],
                _marker: PhantomData,
            },
        ];
        let scalars = [Fq::from(7), Fq::from(11)];

        assert_eq!(
            polynomial_inner_product(&polynomial_views(&polys), scalars.into_iter()),
            inner_product(&polys, scalars.into_iter())
        );
    }
}
