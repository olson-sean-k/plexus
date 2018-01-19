//! Geometric traits and primitives.
//!
//! These traits are used to support high-order operations in generators and
//! `Mesh`es. To use types as geometry in a `Mesh` only requires implementing
//! the `Geometry` and `Attribute` traits. Operation and convertion traits are
//! optional, but enable additional features.

use num::{self, Num, NumCast};

pub mod convert;
pub mod ops;

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

#[derive(Copy, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Triplet<T>(pub T, pub T, pub T);

impl<T> Attribute for Duplet<T>
where
    T: Clone,
{
}

impl<T> Attribute for Triplet<T>
where
    T: Clone,
{
}

impl<T> Geometry for Duplet<T>
where
    T: Clone,
{
    type Vertex = Self;
    type Edge = ();
    type Face = ();
}

impl<T> Geometry for Triplet<T>
where
    T: Clone,
{
    type Vertex = Self;
    type Edge = ();
    type Face = ();
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

#[cfg(feature = "geometry-cgmath")]
mod feature_geometry_cgmath {
    use cgmath::{BaseNum, Point2, Point3, Vector2, Vector3};
    use num::{NumCast, ToPrimitive};

    use geometry::*;

    impl<T, U> From<Point2<U>> for Duplet<T>
    where
        T: NumCast,
        U: BaseNum + ToPrimitive,
    {
        fn from(other: Point2<U>) -> Self {
            Duplet(T::from(other.x).unwrap(), T::from(other.y).unwrap())
        }
    }

    impl<T, U> From<Vector2<U>> for Duplet<T>
    where
        T: NumCast,
        U: BaseNum + ToPrimitive,
    {
        fn from(other: Vector2<U>) -> Self {
            Duplet(T::from(other.x).unwrap(), T::from(other.y).unwrap())
        }
    }

    impl<T, U> Into<Point2<T>> for Duplet<U>
    where
        T: BaseNum + NumCast,
        U: ToPrimitive,
    {
        fn into(self) -> Point2<T> {
            Point2::new(T::from(self.0).unwrap(), T::from(self.1).unwrap())
        }
    }

    impl<T, U> Into<Vector2<T>> for Duplet<U>
    where
        T: BaseNum + NumCast,
        U: ToPrimitive,
    {
        fn into(self) -> Vector2<T> {
            Vector2::new(T::from(self.0).unwrap(), T::from(self.1).unwrap())
        }
    }

    impl<T, U> From<Point3<U>> for Triplet<T>
    where
        T: NumCast,
        U: BaseNum + ToPrimitive,
    {
        fn from(other: Point3<U>) -> Self {
            Triplet(
                T::from(other.x).unwrap(),
                T::from(other.y).unwrap(),
                T::from(other.z).unwrap(),
            )
        }
    }

    impl<T, U> From<Vector3<U>> for Triplet<T>
    where
        T: NumCast,
        U: BaseNum + ToPrimitive,
    {
        fn from(other: Vector3<U>) -> Self {
            Triplet(
                T::from(other.x).unwrap(),
                T::from(other.y).unwrap(),
                T::from(other.z).unwrap(),
            )
        }
    }

    impl<T, U> Into<Point3<T>> for Triplet<U>
    where
        T: BaseNum + NumCast,
        U: ToPrimitive,
    {
        fn into(self) -> Point3<T> {
            Point3::new(
                T::from(self.0).unwrap(),
                T::from(self.1).unwrap(),
                T::from(self.2).unwrap(),
            )
        }
    }

    impl<T, U> Into<Vector3<T>> for Triplet<U>
    where
        T: BaseNum + NumCast,
        U: ToPrimitive,
    {
        fn into(self) -> Vector3<T> {
            Vector3::new(
                T::from(self.0).unwrap(),
                T::from(self.1).unwrap(),
                T::from(self.2).unwrap(),
            )
        }
    }

    impl<T> Attribute for Point2<T>
    where
        T: Clone,
    {
    }

    impl<T> Attribute for Point3<T>
    where
        T: Clone,
    {
    }

    impl<T> Geometry for Point2<T>
    where
        T: Clone,
    {
        type Vertex = Self;
        type Edge = ();
        type Face = ();
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
    use num::{NumCast, ToPrimitive};

    use geometry::*;

    impl<T, U> From<Point2<U>> for Duplet<T>
    where
        T: NumCast,
        U: Scalar + ToPrimitive,
    {
        fn from(other: Point2<U>) -> Self {
            Duplet(T::from(other.x).unwrap(), T::from(other.y).unwrap())
        }
    }

    impl<T, U> From<Vector2<U>> for Duplet<T>
    where
        T: NumCast,
        U: Scalar + ToPrimitive,
    {
        fn from(other: Vector2<U>) -> Self {
            Duplet(T::from(other.x).unwrap(), T::from(other.y).unwrap())
        }
    }

    impl<T, U> Into<Point2<T>> for Duplet<U>
    where
        T: NumCast + Scalar,
        U: ToPrimitive,
    {
        fn into(self) -> Point2<T> {
            Point2::new(T::from(self.0).unwrap(), T::from(self.1).unwrap())
        }
    }

    impl<T, U> Into<Vector2<T>> for Duplet<U>
    where
        T: NumCast + Scalar,
        U: ToPrimitive,
    {
        fn into(self) -> Vector2<T> {
            Vector2::new(T::from(self.0).unwrap(), T::from(self.1).unwrap())
        }
    }

    impl<T, U> From<Point3<U>> for Triplet<T>
    where
        T: NumCast,
        U: Scalar + ToPrimitive,
    {
        fn from(other: Point3<U>) -> Self {
            Triplet(
                T::from(other.x).unwrap(),
                T::from(other.y).unwrap(),
                T::from(other.z).unwrap(),
            )
        }
    }

    impl<T, U> From<Vector3<U>> for Triplet<T>
    where
        T: NumCast,
        U: Scalar + ToPrimitive,
    {
        fn from(other: Vector3<U>) -> Self {
            Triplet(
                T::from(other.x).unwrap(),
                T::from(other.y).unwrap(),
                T::from(other.z).unwrap(),
            )
        }
    }

    impl<T, U> Into<Point3<T>> for Triplet<U>
    where
        T: NumCast + Scalar,
        U: ToPrimitive,
    {
        fn into(self) -> Point3<T> {
            Point3::new(
                T::from(self.0).unwrap(),
                T::from(self.1).unwrap(),
                T::from(self.2).unwrap(),
            )
        }
    }

    impl<T, U> Into<Vector3<T>> for Triplet<U>
    where
        T: NumCast + Scalar,
        U: ToPrimitive,
    {
        fn into(self) -> Vector3<T> {
            Vector3::new(
                T::from(self.0).unwrap(),
                T::from(self.1).unwrap(),
                T::from(self.2).unwrap(),
            )
        }
    }

    impl<T> Attribute for Point2<T>
    where
        T: Scalar,
    {
    }

    impl<T> Attribute for Point3<T>
    where
        T: Scalar,
    {
    }

    impl<T> Geometry for Point2<T>
    where
        T: Scalar,
    {
        type Vertex = Self;
        type Edge = ();
        type Face = ();
    }

    impl<T> Geometry for Point3<T>
    where
        T: Scalar,
    {
        type Vertex = Self;
        type Edge = ();
        type Face = ();
    }
}
