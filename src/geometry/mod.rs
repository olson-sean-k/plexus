use num::{self, Float, Num, NumCast};

use ordered::{HashConjugate, NotNan};

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
mod feature_geometry_cgmath {}

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
