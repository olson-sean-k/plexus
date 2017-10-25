//! Geometric traits and primitives.
//!
//! These traits are used to support high-order operations in generators and
//! `Mesh`es. To use types as geometry in a `Mesh` only requires implementing
//! the `Geometry` and `Attribute` traits. Operation and convertion traits are
//! optional, but enable additional features.

use decorum::Finite;
use num::{self, Float, Num, NumCast};

use ordered::HashConjugate;

pub mod convert;
pub mod ops;

// TODO: `Finite` is used as a hash conjugate and is convertible in geometric
//       types. It may also be good to support `NotNan` for geometric
//       conversions. Is there a way to do that generically (without copying
//       the `Finite` implementations)?

pub trait Attribute: Clone {}

pub trait Geometry: Sized {
    type Vertex: Attribute;
    type Edge: Attribute + Default;
    type Face: Attribute + Default;
}

impl Attribute for () {}

impl Geometry for () {
    type Vertex = ();
    type Edge = ();
    type Face = ();
}

#[derive(Copy, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Duplet<T>(pub T, pub T);

impl<T> HashConjugate for Duplet<T>
where
    T: Float,
{
    type Hash = Duplet<Finite<T>>;

    fn into_hash(self) -> Self::Hash {
        Duplet(
            Finite::from_raw_float(self.0).unwrap(),
            Finite::from_raw_float(self.1).unwrap(),
        )
    }

    fn from_hash(hash: Self::Hash) -> Self {
        Duplet((hash.0).into_raw_float(), (hash.1).into_raw_float())
    }
}

#[derive(Copy, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Triplet<T>(pub T, pub T, pub T);

impl<T> Attribute for Triplet<T>
where
    T: Clone,
{
}

impl<T> Geometry for Triplet<T>
where
    T: Clone,
{
    type Vertex = Self;
    type Edge = ();
    type Face = ();
}

impl<T> HashConjugate for Triplet<T>
where
    T: Float,
{
    type Hash = Triplet<Finite<T>>;

    fn into_hash(self) -> Self::Hash {
        Triplet(
            Finite::from_raw_float(self.0).unwrap(),
            Finite::from_raw_float(self.1).unwrap(),
            Finite::from_raw_float(self.2).unwrap(),
        )
    }

    fn from_hash(hash: Self::Hash) -> Self {
        Triplet(
            (hash.0).into_raw_float(),
            (hash.1).into_raw_float(),
            (hash.2).into_raw_float(),
        )
    }
}

pub fn lerp<T>(a: T, b: T, f: f64) -> T
where
    T: Num + NumCast,
{
    let f = num::clamp(f, 0.0, 1.0);
    let af = <f64 as NumCast>::from(a).unwrap() * (1.0 - f);
    let bf = <f64 as NumCast>::from(b).unwrap() * f;
    <T as NumCast>::from(af + bf).unwrap()
}

// TODO: `Finite` cannot be used as scalar values with cgmath types, because it
//       does not implement `Float` and therefore cannot implement `BaseFloat`.
//       This means cgmath does not support `HashConjugate`.
//
//       The decorum crate could probably provide a hashable type that has no
//       constraints on its value.
#[cfg(feature = "geometry-cgmath")]
mod feature_geometry_cgmath {
    use cgmath::{BaseNum, Point2, Point3, Vector2, Vector3};

    use geometry::*;

    impl<T> From<Point2<T>> for Duplet<T> {
        fn from(other: Point2<T>) -> Self {
            Duplet(other.x, other.y)
        }
    }

    impl<T> From<Vector2<T>> for Duplet<T> {
        fn from(other: Vector2<T>) -> Self {
            Duplet(other.x, other.y)
        }
    }

    impl<T> Into<Point2<T>> for Duplet<T>
    where
        T: BaseNum,
    {
        fn into(self) -> Point2<T> {
            Point2::new(self.0, self.1)
        }
    }

    impl<T> Into<Vector2<T>> for Duplet<T>
    where
        T: BaseNum,
    {
        fn into(self) -> Vector2<T> {
            Vector2::new(self.0, self.1)
        }
    }

    impl<T> From<Point3<T>> for Triplet<T> {
        fn from(other: Point3<T>) -> Self {
            Triplet(other.x, other.y, other.z)
        }
    }

    impl<T> From<Vector3<T>> for Triplet<T> {
        fn from(other: Vector3<T>) -> Self {
            Triplet(other.x, other.y, other.z)
        }
    }

    impl<T> Into<Point3<T>> for Triplet<T>
    where
        T: BaseNum,
    {
        fn into(self) -> Point3<T> {
            Point3::new(self.0, self.1, self.2)
        }
    }

    impl<T> Into<Vector3<T>> for Triplet<T>
    where
        T: BaseNum,
    {
        fn into(self) -> Vector3<T> {
            Vector3::new(self.0, self.1, self.2)
        }
    }

    impl<T> Attribute for Point3<T>
    where
        T: Clone,
    {
    }

    impl<T> Geometry for Point3<T>
    where
        T: Clone,
    {
        type Vertex = Self;
        type Edge = ();
        type Face = ();
    }
}

#[cfg(feature = "geometry-nalgebra")]
mod feature_geometry_nalgebra {
    use nalgebra::{Point2, Point3, Scalar, Vector2, Vector3};

    use geometry::*;

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

    impl<T> Attribute for Point3<T>
    where
        T: Scalar,
    {
    }

    impl<T> Geometry for Point3<T>
    where
        T: Scalar,
    {
        type Vertex = Self;
        type Edge = ();
        type Face = ();
    }

    impl<T> HashConjugate for Point2<T>
    where
        T: Float + Scalar,
    {
        type Hash = Point2<Finite<T>>;

        fn into_hash(self) -> Self::Hash {
            Point2::new(
                Finite::from_raw_float(self.x).unwrap(),
                Finite::from_raw_float(self.y).unwrap(),
            )
        }

        fn from_hash(hash: Self::Hash) -> Self {
            Point2::new(hash.x.into_raw_float(), hash.y.into_raw_float())
        }
    }

    impl<T> HashConjugate for Point3<T>
    where
        T: Float + Scalar,
    {
        type Hash = Point3<Finite<T>>;

        fn into_hash(self) -> Self::Hash {
            Point3::new(
                Finite::from_raw_float(self.x).unwrap(),
                Finite::from_raw_float(self.y).unwrap(),
                Finite::from_raw_float(self.z).unwrap(),
            )
        }

        fn from_hash(hash: Self::Hash) -> Self {
            Point3::new(
                hash.x.into_raw_float(),
                hash.y.into_raw_float(),
                hash.z.into_raw_float(),
            )
        }
    }

    impl<T> HashConjugate for Vector2<T>
    where
        T: Float + Scalar,
    {
        type Hash = Vector2<Finite<T>>;

        fn into_hash(self) -> Self::Hash {
            Vector2::new(
                Finite::from_raw_float(self.x).unwrap(),
                Finite::from_raw_float(self.y).unwrap(),
            )
        }

        fn from_hash(hash: Self::Hash) -> Self {
            Vector2::new(hash.x.into_raw_float(), hash.y.into_raw_float())
        }
    }

    impl<T> HashConjugate for Vector3<T>
    where
        T: Float + Scalar,
    {
        type Hash = Vector3<Finite<T>>;

        fn into_hash(self) -> Self::Hash {
            Vector3::new(
                Finite::from_raw_float(self.x).unwrap(),
                Finite::from_raw_float(self.y).unwrap(),
                Finite::from_raw_float(self.z).unwrap(),
            )
        }

        fn from_hash(hash: Self::Hash) -> Self {
            Vector3::new(
                hash.x.into_raw_float(),
                hash.y.into_raw_float(),
                hash.z.into_raw_float(),
            )
        }
    }
}
