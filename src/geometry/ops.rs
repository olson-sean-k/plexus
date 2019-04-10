use decorum::{Real, R64};
use num::{Num, NumCast, One};

use crate::geometry::space::{AbstractSpace, VectorSpace};
use crate::geometry::{self, Duplet, Half, Triplet};

pub trait Normalize {
    fn normalize(self) -> Self;
}

impl<T> Normalize for T
where
    T: Dot<Output = <T as AbstractSpace>::Scalar> + VectorSpace,
    T::Scalar: Real,
{
    fn normalize(self) -> Self {
        self.clone() * (T::Scalar::one() / self.magnitude())
    }
}

pub trait Project<T = Self> {
    type Output;

    fn project(self, other: T) -> Self::Output;
}

impl<T> Project<T> for T
where
    T: Dot<Output = <T as AbstractSpace>::Scalar> + VectorSpace,
    T::Scalar: Real,
{
    type Output = T;

    fn project(self, other: T) -> Self::Output {
        let n = other.dot(self.clone());
        let d = self.clone().dot(self.clone());
        self * (n / d)
    }
}

pub trait Magnitude: Sized {
    type Output;

    fn square_magnitude(self) -> Self::Output;

    fn magnitude(self) -> Self::Output;
}

impl<T> Magnitude for T
where
    T: Dot<Output = <T as AbstractSpace>::Scalar> + VectorSpace,
    T::Scalar: Real,
{
    type Output = <T as Dot>::Output;

    fn square_magnitude(self) -> Self::Output {
        Dot::dot(self.clone(), self)
    }

    fn magnitude(self) -> Self::Output {
        Real::sqrt(self.square_magnitude())
    }
}

pub trait Interpolate<T = Self>: Sized {
    type Output;

    fn lerp(self, other: T, f: R64) -> Self::Output;

    fn midpoint(self, other: T) -> Self::Output {
        self.lerp(other, Half::half())
    }
}

pub trait Dot<T = Self> {
    type Output;

    fn dot(self, other: T) -> Self::Output;
}

pub trait Cross<T = Self> {
    type Output;

    fn cross(self, other: T) -> Self::Output;
}

impl<T> Interpolate for Duplet<T>
where
    T: Copy + Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        Duplet(
            geometry::lerp(self.0, other.0, f),
            geometry::lerp(self.1, other.1, f),
        )
    }
}

impl<T> Interpolate for Triplet<T>
where
    T: Copy + Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        Triplet(
            geometry::lerp(self.0, other.0, f),
            geometry::lerp(self.1, other.1, f),
            geometry::lerp(self.2, other.2, f),
        )
    }
}
