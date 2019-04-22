use num::{Num, NumCast, One, Zero};
use std::ops::{Add, Mul, Neg, Sub};

use crate::geometry::ops::{Dot, Magnitude, Normalize, Project};

pub trait AbstractSpace: Clone {
    type Scalar: Clone + Neg<Output = Self::Scalar> + Num + NumCast;
}

pub trait VectorSpace:
    AbstractSpace
    + Add<Output = Self>
    + Mul<<Self as AbstractSpace>::Scalar, Output = Self>
    + Neg<Output = Self>
    + Zero
{
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
            NumCast::from(n).map(move |n| sum * (<Self as AbstractSpace>::Scalar::one() / n))
        }
        else {
            None
        }
    }
}

impl<T> VectorSpace for T where
    T: AbstractSpace
        + Add<Output = T>
        + Mul<<T as AbstractSpace>::Scalar, Output = T>
        + Neg<Output = T>
        + Zero
{
}

pub trait InnerSpace:
    Dot<Output = <Self as AbstractSpace>::Scalar>
    + Magnitude<Output = <Self as AbstractSpace>::Scalar>
    + Normalize
    + Project<Output = Self>
    + VectorSpace
{
}

impl<T> InnerSpace for T where
    T: Dot<Output = <T as AbstractSpace>::Scalar>
        + Magnitude<Output = <T as AbstractSpace>::Scalar>
        + Normalize
        + Project<Output = Self>
        + VectorSpace
{
}

pub trait Origin {
    fn origin() -> Self;
}

pub trait EuclideanSpace:
    Add<<Self as EuclideanSpace>::Difference, Output = Self>
    + Clone
    + Origin
    + Sub<Output = <Self as EuclideanSpace>::Difference>
{
    type Difference: InnerSpace;

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
