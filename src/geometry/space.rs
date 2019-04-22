use num::{Num, NumCast, One, Zero};
use std::ops::{Add, Mul, Neg, Sub};

use crate::geometry::ops::{Dot, Magnitude, Normalize, Project};

pub trait VectorSpace:
    Add<Output = Self>
    + Clone
    + Mul<<Self as VectorSpace>::Scalar, Output = Self>
    + Neg<Output = Self>
    + Zero
{
    type Scalar: Clone + Neg<Output = Self::Scalar> + Num + NumCast;

    fn mean<I>(vectors: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self>,
    {
        let mut vectors = vectors.into_iter();
        if let Some(mut sum) = vectors.next() {
            let mut n = 1usize;
            for vector in vectors {
                n += 1;
                sum = sum + vector;
            }
            NumCast::from(n).map(move |n| sum * (Self::Scalar::one() / n))
        }
        else {
            None
        }
    }
}

pub trait InnerSpace:
    Dot<Output = <Self as VectorSpace>::Scalar>
    + Magnitude<Output = <Self as VectorSpace>::Scalar>
    + Normalize
    + Project<Output = Self>
    + VectorSpace
{
}

impl<T> InnerSpace for T where
    T: Dot<Output = <T as VectorSpace>::Scalar>
        + Magnitude<Output = <T as VectorSpace>::Scalar>
        + Normalize
        + Project<Output = Self>
        + VectorSpace
{
}

pub trait EuclideanSpace:
    Add<<Self as EuclideanSpace>::Difference, Output = Self>
    + Clone
    + Sub<Output = <Self as EuclideanSpace>::Difference>
{
    type Difference: InnerSpace;

    fn origin() -> Self;

    fn coordinates(&self) -> Self::Difference {
        self.clone() - Self::origin()
    }

    fn centroid<I>(points: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self>,
    {
        VectorSpace::mean(points.into_iter().map(|point| point.coordinates()))
            .map(|mean| Self::origin() + mean)
    }
}
