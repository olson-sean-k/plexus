use decorum::{Ordered, Primitive};
use num::Float;
use std::hash::Hash;

use geometry::{Duplet, Triplet};

pub trait FromGeometry<T> {
    fn from_geometry(other: T) -> Self;
}

pub trait IntoGeometry<T> {
    fn into_geometry(self) -> T;
}

impl<T, U> IntoGeometry<U> for T
where
    U: FromGeometry<T>,
{
    fn into_geometry(self) -> U {
        U::from_geometry(self)
    }
}

impl<T> FromGeometry<T> for T {
    fn from_geometry(other: T) -> Self {
        other
    }
}

// TODO: The interior versions of geometry conversion traits do not have a
//       reflexive implementation. This allows for conversions from `Mesh<T>`
//       to `Mesh<U>`, where `T` and `U` may be the same.
//
//       This is a bit confusing; consider removing these if they aren't
//       useful.
pub trait FromInteriorGeometry<T> {
    fn from_interior_geometry(other: T) -> Self;
}

pub trait IntoInteriorGeometry<T> {
    fn into_interior_geometry(self) -> T;
}

impl<T, U> IntoInteriorGeometry<U> for T
where
    U: FromInteriorGeometry<T>,
{
    fn into_interior_geometry(self) -> U {
        U::from_interior_geometry(self)
    }
}

pub trait AsPosition {
    type Target;

    fn as_position(&self) -> &Self::Target;
    fn as_position_mut(&mut self) -> &mut Self::Target;
}

/// Provides conversion to and from a conjugate type that can be hashed.
///
/// This trait is primarily used to convert geometry to and from hashable data
/// for indexing.
pub trait HashConjugate: Sized {
    /// Conjugate type that provides `Eq` and `Hash` implementations.
    type Hash: Eq + Hash;

    /// Converts into the conjugate type.
    fn into_hash(self) -> Self::Hash;

    /// Converts from the conjugate type.
    fn from_hash(hash: Self::Hash) -> Self;
}

impl<T> HashConjugate for Duplet<T>
where
    T: Float + Primitive,
{
    type Hash = Duplet<Ordered<T>>;

    fn into_hash(self) -> Self::Hash {
        Duplet(
            Ordered::from_raw_float(self.0),
            Ordered::from_raw_float(self.1),
        )
    }

    fn from_hash(hash: Self::Hash) -> Self {
        Duplet((hash.0).into_raw_float(), (hash.1).into_raw_float())
    }
}

impl<T> HashConjugate for Triplet<T>
where
    T: Float + Primitive,
{
    type Hash = Triplet<Ordered<T>>;

    fn into_hash(self) -> Self::Hash {
        Triplet(
            Ordered::from_raw_float(self.0),
            Ordered::from_raw_float(self.1),
            Ordered::from_raw_float(self.2),
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

// TODO: Implement `FromGeometry` for more proxy types than just `Finite`
//       (using specialization or otherwise).
// TODO: Implement `FromGeometry` for points and vectors with `FloatProxy`
//       scalars once specialization lands. This isn't possible now, because it
//       would conflict with the blanket implementation for some shared scalar
//       type `T`, which is arguably a more important implementation.
#[cfg(feature = "geometry-cgmath")]
mod feature_geometry_cgmath {
    use cgmath::{BaseFloat, BaseNum, Point2, Point3, Vector2, Vector3};
    use decorum::{Finite, Ordered, Primitive};

    use geometry::{Duplet, Triplet};
    use geometry::convert::*;

    impl<T> FromGeometry<Duplet<T>> for Point2<T>
    where
        T: BaseNum,
    {
        fn from_geometry(other: Duplet<T>) -> Self {
            Point2::new(other.0, other.1)
        }
    }

    impl<T> FromGeometry<Triplet<T>> for Point3<T>
    where
        T: BaseNum,
    {
        fn from_geometry(other: Triplet<T>) -> Self {
            Point3::new(other.0, other.1, other.2)
        }
    }

    impl<T> FromGeometry<Triplet<Finite<T>>> for Point3<T>
    where
        T: BaseFloat + BaseNum + Primitive,
    {
        fn from_geometry(other: Triplet<Finite<T>>) -> Self {
            Point3::new(
                other.0.into_raw_float(),
                other.1.into_raw_float(),
                other.2.into_raw_float(),
            )
        }
    }

    impl<T> FromGeometry<Point3<Finite<T>>> for Point3<T>
    where
        T: BaseFloat + BaseNum + Primitive,
    {
        fn from_geometry(other: Point3<Finite<T>>) -> Self {
            Point3::new(
                other.x.into_raw_float(),
                other.y.into_raw_float(),
                other.z.into_raw_float(),
            )
        }
    }

    impl<T> FromGeometry<Duplet<T>> for Vector2<T>
    where
        T: BaseNum,
    {
        fn from_geometry(other: Duplet<T>) -> Self {
            Vector2::new(other.0, other.1)
        }
    }

    impl<T> FromGeometry<Triplet<T>> for Vector3<T>
    where
        T: BaseNum,
    {
        fn from_geometry(other: Triplet<T>) -> Self {
            Vector3::new(other.0, other.1, other.2)
        }
    }

    impl<T> AsPosition for Point3<T>
    where
        T: BaseNum,
    {
        type Target = Self;

        fn as_position(&self) -> &Self::Target {
            self
        }

        fn as_position_mut(&mut self) -> &mut Self::Target {
            self
        }
    }

    impl<T> HashConjugate for Point2<T>
    where
        T: BaseFloat + Float + Primitive,
    {
        type Hash = Point2<Ordered<T>>;

        fn into_hash(self) -> Self::Hash {
            Point2::new(
                Ordered::from_raw_float(self.x),
                Ordered::from_raw_float(self.y),
            )
        }

        fn from_hash(hash: Self::Hash) -> Self {
            Point2::new(hash.x.into_raw_float(), hash.y.into_raw_float())
        }
    }

    impl<T> HashConjugate for Point3<T>
    where
        T: BaseFloat + Float + Primitive,
    {
        type Hash = Point3<Ordered<T>>;

        fn into_hash(self) -> Self::Hash {
            Point3::new(
                Ordered::from_raw_float(self.x),
                Ordered::from_raw_float(self.y),
                Ordered::from_raw_float(self.z),
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
        T: BaseFloat + Float + Primitive,
    {
        type Hash = Vector2<Ordered<T>>;

        fn into_hash(self) -> Self::Hash {
            Vector2::new(
                Ordered::from_raw_float(self.x),
                Ordered::from_raw_float(self.y),
            )
        }

        fn from_hash(hash: Self::Hash) -> Self {
            Vector2::new(hash.x.into_raw_float(), hash.y.into_raw_float())
        }
    }

    impl<T> HashConjugate for Vector3<T>
    where
        T: BaseFloat + Float + Primitive,
    {
        type Hash = Vector3<Ordered<T>>;

        fn into_hash(self) -> Self::Hash {
            Vector3::new(
                Ordered::from_raw_float(self.x),
                Ordered::from_raw_float(self.y),
                Ordered::from_raw_float(self.z),
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

#[cfg(feature = "geometry-nalgebra")]
mod feature_geometry_nalgebra {
    use decorum::{Finite, Ordered, Primitive};
    use nalgebra::{Point2, Point3, Scalar, Vector2, Vector3};
    use num::Float;

    use geometry::{Duplet, Triplet};
    use geometry::convert::*;

    impl<T> FromGeometry<Duplet<T>> for Point2<T>
    where
        T: Scalar,
    {
        fn from_geometry(other: Duplet<T>) -> Self {
            Point2::new(other.0, other.1)
        }
    }

    impl<T> FromGeometry<Triplet<T>> for Point3<T>
    where
        T: Scalar,
    {
        fn from_geometry(other: Triplet<T>) -> Self {
            Point3::new(other.0, other.1, other.2)
        }
    }

    impl<T> FromGeometry<Triplet<Finite<T>>> for Point3<T>
    where
        T: Float + Primitive + Scalar,
    {
        fn from_geometry(other: Triplet<Finite<T>>) -> Self {
            Point3::new(
                other.0.into_raw_float(),
                other.1.into_raw_float(),
                other.2.into_raw_float(),
            )
        }
    }

    impl<T> FromGeometry<Point3<Finite<T>>> for Point3<T>
    where
        T: Float + Primitive + Scalar,
    {
        fn from_geometry(other: Point3<Finite<T>>) -> Self {
            Point3::new(
                other.x.into_raw_float(),
                other.y.into_raw_float(),
                other.z.into_raw_float(),
            )
        }
    }

    impl<T> FromGeometry<Duplet<T>> for Vector2<T>
    where
        T: Scalar,
    {
        fn from_geometry(other: Duplet<T>) -> Self {
            Vector2::new(other.0, other.1)
        }
    }

    impl<T> FromGeometry<Triplet<T>> for Vector3<T>
    where
        T: Scalar,
    {
        fn from_geometry(other: Triplet<T>) -> Self {
            Vector3::new(other.0, other.1, other.2)
        }
    }

    impl<T> AsPosition for Point3<T>
    where
        T: Scalar,
    {
        type Target = Self;

        fn as_position(&self) -> &Self::Target {
            self
        }

        fn as_position_mut(&mut self) -> &mut Self::Target {
            self
        }
    }

    impl<T> HashConjugate for Point2<T>
    where
        T: Float + Primitive + Scalar,
    {
        type Hash = Point2<Ordered<T>>;

        fn into_hash(self) -> Self::Hash {
            Point2::new(
                Ordered::from_raw_float(self.x),
                Ordered::from_raw_float(self.y),
            )
        }

        fn from_hash(hash: Self::Hash) -> Self {
            Point2::new(hash.x.into_raw_float(), hash.y.into_raw_float())
        }
    }

    impl<T> HashConjugate for Point3<T>
    where
        T: Float + Primitive + Scalar,
    {
        type Hash = Point3<Ordered<T>>;

        fn into_hash(self) -> Self::Hash {
            Point3::new(
                Ordered::from_raw_float(self.x),
                Ordered::from_raw_float(self.y),
                Ordered::from_raw_float(self.z),
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
        T: Float + Primitive + Scalar,
    {
        type Hash = Vector2<Ordered<T>>;

        fn into_hash(self) -> Self::Hash {
            Vector2::new(
                Ordered::from_raw_float(self.x),
                Ordered::from_raw_float(self.y),
            )
        }

        fn from_hash(hash: Self::Hash) -> Self {
            Vector2::new(hash.x.into_raw_float(), hash.y.into_raw_float())
        }
    }

    impl<T> HashConjugate for Vector3<T>
    where
        T: Float + Primitive + Scalar,
    {
        type Hash = Vector3<Ordered<T>>;

        fn into_hash(self) -> Self::Hash {
            Vector3::new(
                Ordered::from_raw_float(self.x),
                Ordered::from_raw_float(self.y),
                Ordered::from_raw_float(self.z),
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
