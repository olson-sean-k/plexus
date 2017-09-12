use num::{self, Float, Num, NumCast};
use std::hash::Hash;

use NotNan;

pub trait Unit: Copy + Num {
    fn unit_radius() -> (Self, Self);
    fn unit_width() -> (Self, Self);
}

// TODO: https://github.com/reem/rust-ordered-float/pull/31
//       Once `NotNan` implements numeric traits, implement `Unit` for `NotNan`
//       and allow it to be used as vertex geometry in generators.
macro_rules! unit {
    (integer => $($t:ty),*) => {$(
        impl Unit for $t {
            fn unit_radius() -> (Self, Self) {
                use num::{One, Zero};
                (Self::zero(), Self::one() + Self::one())
            }

            fn unit_width() -> (Self, Self) {
                use num::{One, Zero};
                (Self::zero(), Self::one())
            }
        }
    )*};
    (real => $($t:ty),*) => {$(
        impl Unit for $t {
            fn unit_radius() -> (Self, Self) {
                use num::One;
                (-Self::one(), Self::one())
            }

            fn unit_width() -> (Self, Self) {
                use num::One;
                let half = Self::one() / (Self::one() + Self::one());
                (-half, half)
            }
        }
    )*};
}
unit!(integer => i8, i16, i32, i64, u8, u16, u32, u64);
unit!(real => f32, f64);

pub trait HashConjugate: Sized {
    type Hash: Eq + Hash;

    fn into_hash(self) -> Self::Hash;
    fn from_hash(hash: Self::Hash) -> Self;
}

impl<T> HashConjugate for (T, T)
where
    T: Float + Unit,
{
    type Hash = (NotNan<T>, NotNan<T>);

    fn into_hash(self) -> Self::Hash {
        (NotNan::new(self.0).unwrap(), NotNan::new(self.1).unwrap())
    }

    fn from_hash(hash: Self::Hash) -> Self {
        ((hash.0).into_inner(), (hash.1).into_inner())
    }
}

impl<T> HashConjugate for (T, T, T)
where
    T: Float + Unit,
{
    type Hash = (NotNan<T>, NotNan<T>, NotNan<T>);

    fn into_hash(self) -> Self::Hash {
        (
            NotNan::new(self.0).unwrap(),
            NotNan::new(self.1).unwrap(),
            NotNan::new(self.2).unwrap(),
        )
    }

    fn from_hash(hash: Self::Hash) -> Self {
        (
            (hash.0).into_inner(),
            (hash.1).into_inner(),
            (hash.2).into_inner(),
        )
    }
}

pub trait Interpolate<T = Self>: Sized {
    type Output;

    fn lerp(self, other: T, f: f64) -> Self::Output;

    fn midpoint(self, other: T) -> Self::Output {
        self.lerp(other, 0.5)
    }
}

impl<T> Interpolate for (T, T)
where
    T: Copy + Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: f64) -> Self::Output {
        (lerp(self.0, other.0, f), lerp(self.1, other.1, f))
    }
}

impl<T> Interpolate for (T, T, T)
where
    T: Copy + Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: f64) -> Self::Output {
        (
            lerp(self.0, other.0, f),
            lerp(self.1, other.1, f),
            lerp(self.2, other.2, f),
        )
    }
}

fn lerp<T>(a: T, b: T, f: f64) -> T
where
    T: Num + NumCast,
{
    let f = num::clamp(f, 0.0, 1.0);
    let af = <f64 as NumCast>::from(a).unwrap() * (1.0 - f);
    let bf = <f64 as NumCast>::from(b).unwrap() * f;
    <T as NumCast>::from(af + bf).unwrap()
}

#[cfg(feature = "geometry-nalgebra")]
mod feature {
    use nalgebra::{Point2, Point3, Scalar, Vector2, Vector3};

    use super::*;

    impl<T> HashConjugate for Point2<T>
    where
        T: Float + Scalar + Unit,
    {
        type Hash = (NotNan<T>, NotNan<T>);

        fn into_hash(self) -> Self::Hash {
            (NotNan::new(self.x).unwrap(), NotNan::new(self.y).unwrap())
        }

        fn from_hash(hash: Self::Hash) -> Self {
            Point2::new((hash.0).into_inner(), (hash.1).into_inner())
        }
    }

    impl<T> HashConjugate for Point3<T>
    where
        T: Float + Scalar + Unit,
    {
        type Hash = (NotNan<T>, NotNan<T>, NotNan<T>);

        fn into_hash(self) -> Self::Hash {
            (
                NotNan::new(self.x).unwrap(),
                NotNan::new(self.y).unwrap(),
                NotNan::new(self.z).unwrap(),
            )
        }

        fn from_hash(hash: Self::Hash) -> Self {
            Point3::new(
                (hash.0).into_inner(),
                (hash.1).into_inner(),
                (hash.2).into_inner(),
            )
        }
    }

    impl<T> HashConjugate for Vector2<T>
    where
        T: Float + Scalar + Unit,
    {
        type Hash = (NotNan<T>, NotNan<T>);

        fn into_hash(self) -> Self::Hash {
            (NotNan::new(self.x).unwrap(), NotNan::new(self.y).unwrap())
        }

        fn from_hash(hash: Self::Hash) -> Self {
            Vector2::new((hash.0).into_inner(), (hash.1).into_inner())
        }
    }

    impl<T> HashConjugate for Vector3<T>
    where
        T: Float + Scalar + Unit,
    {
        type Hash = (NotNan<T>, NotNan<T>, NotNan<T>);

        fn into_hash(self) -> Self::Hash {
            (
                NotNan::new(self.x).unwrap(),
                NotNan::new(self.y).unwrap(),
                NotNan::new(self.z).unwrap(),
            )
        }

        fn from_hash(hash: Self::Hash) -> Self {
            Vector3::new(
                (hash.0).into_inner(),
                (hash.1).into_inner(),
                (hash.2).into_inner(),
            )
        }
    }

    impl<T> Interpolate for Point2<T>
    where
        T: NumCast + Scalar + Unit,
    {
        type Output = Self;

        fn lerp(self, other: Self, f: f64) -> Self::Output {
            Point2::new(lerp(self.x, other.x, f), lerp(self.y, other.y, f))
        }
    }

    impl<T> Interpolate for Point3<T>
    where
        T: NumCast + Scalar + Unit,
    {
        type Output = Self;

        fn lerp(self, other: Self, f: f64) -> Self::Output {
            Point3::new(
                lerp(self.x, other.x, f),
                lerp(self.y, other.y, f),
                lerp(self.z, other.z, f),
            )
        }
    }

    impl<T> Interpolate for Vector2<T>
    where
        T: NumCast + Scalar + Unit,
    {
        type Output = Self;

        fn lerp(self, other: Self, f: f64) -> Self::Output {
            Vector2::new(lerp(self.x, other.x, f), lerp(self.y, other.y, f))
        }
    }

    impl<T> Interpolate for Vector3<T>
    where
        T: NumCast + Scalar + Unit,
    {
        type Output = Self;

        fn lerp(self, other: Self, f: f64) -> Self::Output {
            Vector3::new(
                lerp(self.x, other.x, f),
                lerp(self.y, other.y, f),
                lerp(self.z, other.z, f),
            )
        }
    }
}

#[cfg(not(feature = "geometry-nalgebra"))]
mod feature {}

pub use self::feature::*;
