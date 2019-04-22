use decorum::Real;
use num::{NumCast, One, Zero};
use std::ops::{Add, Mul, Neg, Sub};

use crate::geometry::ops::{Dot, Magnitude, Normalize, Project};

pub trait VectorSpace:
    Add<Output = Self>
    + Clone
    + Mul<<Self as VectorSpace>::Scalar, Output = Self>
    + Neg<Output = Self>
    + Zero
{
    type Scalar: Real;

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

pub trait InnerSpace: Dot<Output = <Self as VectorSpace>::Scalar> + VectorSpace {}

impl<T> Normalize for T
where
    T: InnerSpace,
{
    fn normalize(self) -> Self {
        self.clone() * (T::Scalar::one() / self.magnitude())
    }
}

impl<T> Project<T> for T
where
    T: InnerSpace,
{
    type Output = T;

    fn project(self, other: T) -> Self::Output {
        let n = other.dot(self.clone());
        let d = self.clone().dot(self.clone());
        self * (n / d)
    }
}

impl<T> Magnitude for T
where
    T: InnerSpace,
{
    type Output = <T as Dot>::Output;

    fn square_magnitude(self) -> Self::Output {
        Dot::dot(self.clone(), self)
    }

    fn magnitude(self) -> Self::Output {
        Real::sqrt(self.square_magnitude())
    }
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
