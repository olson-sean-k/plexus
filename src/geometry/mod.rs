//! Geometric traits and primitives.
//!
//! These traits are used to support high-order operations in generators and
//! `Mesh`es. To use types as geometry in a `Mesh` only requires implementing
//! the `Geometry` and `Attribute` traits. Operation and convertion traits are
//! optional, but enable additional features.

use num::{self, Num, NumCast};

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
}
