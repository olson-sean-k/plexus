use num::{self, Float, Num, NumCast};

use ordered::{HashConjugate, NotNan};

#[derive(Copy, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Duplet<T>(pub T, pub T);

#[derive(Copy, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Triplet<T>(pub T, pub T, pub T);

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

pub trait Interpolate<T = Self>: Sized {
    type Output;

    fn lerp(self, other: T, f: f64) -> Self::Output;

    fn midpoint(self, other: T) -> Self::Output {
        self.lerp(other, 0.5)
    }
}

impl<T> Interpolate for Duplet<T>
where
    T: Copy + Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: f64) -> Self::Output {
        Duplet(lerp(self.0, other.0, f), lerp(self.1, other.1, f))
    }
}

impl<T> Interpolate for Triplet<T>
where
    T: Copy + Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: f64) -> Self::Output {
        Triplet(
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

impl<T> HashConjugate for Duplet<T>
where
    T: Float,
{
    type Hash = Duplet<NotNan<T>>;

    fn into_hash(self) -> Self::Hash {
        Duplet(NotNan::new(self.0).unwrap(), NotNan::new(self.1).unwrap())
    }

    fn from_hash(hash: Self::Hash) -> Self {
        Duplet((hash.0).into_inner(), (hash.1).into_inner())
    }
}

impl<T> HashConjugate for Triplet<T>
where
    T: Float,
{
    type Hash = Triplet<NotNan<T>>;

    fn into_hash(self) -> Self::Hash {
        Triplet(
            NotNan::new(self.0).unwrap(),
            NotNan::new(self.1).unwrap(),
            NotNan::new(self.2).unwrap(),
        )
    }

    fn from_hash(hash: Self::Hash) -> Self {
        Triplet(
            (hash.0).into_inner(),
            (hash.1).into_inner(),
            (hash.2).into_inner(),
        )
    }
}

#[cfg(feature = "geometry-cgmath")]
mod feature_geometry_cgmath {}
#[cfg(not(feature = "geometry-cgmath"))]
mod feature_geometry_cgmath {}

#[cfg(feature = "geometry-nalgebra")]
mod feature_geometry_nalgebra {
    use nalgebra::{Point2, Point3, Scalar, Vector2, Vector3};

    use super::*;

    impl<T> From<Point2<T>> for Duplet<T>
    where
        T: Scalar,
    {
        fn from(other: Point2<T>) -> Self {
            Duplet(other.x, other.y)
        }
    }

    impl<T> From<Vector2<T>> for Duplet<T>
    where
        T: Scalar,
    {
        fn from(other: Vector2<T>) -> Self {
            Duplet(other.x, other.y)
        }
    }

    impl<T> Into<Point2<T>> for Duplet<T>
    where
        T: Scalar,
    {
        fn into(self) -> Point2<T> {
            Point2::new(self.0, self.1)
        }
    }

    impl<T> Into<Vector2<T>> for Duplet<T>
    where
        T: Scalar,
    {
        fn into(self) -> Vector2<T> {
            Vector2::new(self.0, self.1)
        }
    }

    impl<T> From<Point3<T>> for Triplet<T>
    where
        T: Scalar,
    {
        fn from(other: Point3<T>) -> Self {
            Triplet(other.x, other.y, other.z)
        }
    }

    impl<T> From<Vector3<T>> for Triplet<T>
    where
        T: Scalar,
    {
        fn from(other: Vector3<T>) -> Self {
            Triplet(other.x, other.y, other.z)
        }
    }

    impl<T> Into<Point3<T>> for Triplet<T>
    where
        T: Scalar,
    {
        fn into(self) -> Point3<T> {
            Point3::new(self.0, self.1, self.2)
        }
    }

    impl<T> Into<Vector3<T>> for Triplet<T>
    where
        T: Scalar,
    {
        fn into(self) -> Vector3<T> {
            Vector3::new(self.0, self.1, self.2)
        }
    }

    impl<T> Interpolate for Point2<T>
    where
        T: Num + NumCast + Scalar,
    {
        type Output = Self;

        fn lerp(self, other: Self, f: f64) -> Self::Output {
            Point2::new(lerp(self.x, other.x, f), lerp(self.y, other.y, f))
        }
    }

    impl<T> Interpolate for Point3<T>
    where
        T: Num + NumCast + Scalar,
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
        T: Num + NumCast + Scalar,
    {
        type Output = Self;

        fn lerp(self, other: Self, f: f64) -> Self::Output {
            Vector2::new(lerp(self.x, other.x, f), lerp(self.y, other.y, f))
        }
    }

    impl<T> Interpolate for Vector3<T>
    where
        T: Num + NumCast + Scalar,
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

    impl<T> HashConjugate for Point2<T>
    where
        T: Float + Scalar,
    {
        type Hash = Point2<NotNan<T>>;

        fn into_hash(self) -> Self::Hash {
            Point2::new(NotNan::new(self.x).unwrap(), NotNan::new(self.y).unwrap())
        }

        fn from_hash(hash: Self::Hash) -> Self {
            Point2::new(hash.x.into_inner(), hash.y.into_inner())
        }
    }

    impl<T> HashConjugate for Point3<T>
    where
        T: Float + Scalar,
    {
        type Hash = Point3<NotNan<T>>;

        fn into_hash(self) -> Self::Hash {
            Point3::new(
                NotNan::new(self.x).unwrap(),
                NotNan::new(self.y).unwrap(),
                NotNan::new(self.z).unwrap(),
            )
        }

        fn from_hash(hash: Self::Hash) -> Self {
            Point3::new(
                hash.x.into_inner(),
                hash.y.into_inner(),
                hash.z.into_inner(),
            )
        }
    }

    impl<T> HashConjugate for Vector2<T>
    where
        T: Float + Scalar,
    {
        type Hash = Vector2<NotNan<T>>;

        fn into_hash(self) -> Self::Hash {
            Vector2::new(NotNan::new(self.x).unwrap(), NotNan::new(self.y).unwrap())
        }

        fn from_hash(hash: Self::Hash) -> Self {
            Vector2::new(hash.x.into_inner(), hash.y.into_inner())
        }
    }

    impl<T> HashConjugate for Vector3<T>
    where
        T: Float + Scalar,
    {
        type Hash = Vector3<NotNan<T>>;

        fn into_hash(self) -> Self::Hash {
            Vector3::new(
                NotNan::new(self.x).unwrap(),
                NotNan::new(self.y).unwrap(),
                NotNan::new(self.z).unwrap(),
            )
        }

        fn from_hash(hash: Self::Hash) -> Self {
            Vector3::new(
                hash.x.into_inner(),
                hash.y.into_inner(),
                hash.z.into_inner(),
            )
        }
    }
}
#[cfg(not(feature = "geometry-nalgebra"))]
mod feature_geometry_nalgebra {}

pub use self::feature_geometry_cgmath::*;
pub use self::feature_geometry_nalgebra::*;
