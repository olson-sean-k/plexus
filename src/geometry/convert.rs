use decorum::{Ordered, Primitive};
use num::{Float, NumCast, ToPrimitive};
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

impl<T, U> FromGeometry<(U, U)> for Duplet<T>
where
    T: NumCast,
    U: ToPrimitive,
{
    fn from_geometry(other: (U, U)) -> Self {
        Duplet(T::from(other.0).unwrap(), T::from(other.1).unwrap())
    }
}

impl<T, U> FromGeometry<(U, U, U)> for Triplet<T>
where
    T: NumCast,
    U: ToPrimitive,
{
    fn from_geometry(other: (U, U, U)) -> Self {
        Triplet(
            T::from(other.0).unwrap(),
            T::from(other.1).unwrap(),
            T::from(other.2).unwrap(),
        )
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
        Duplet(Ordered::from_inner(self.0), Ordered::from_inner(self.1))
    }

    fn from_hash(hash: Self::Hash) -> Self {
        Duplet((hash.0).into_inner(), (hash.1).into_inner())
    }
}

impl<T> HashConjugate for Triplet<T>
where
    T: Float + Primitive,
{
    type Hash = Triplet<Ordered<T>>;

    fn into_hash(self) -> Self::Hash {
        Triplet(
            Ordered::from_inner(self.0),
            Ordered::from_inner(self.1),
            Ordered::from_inner(self.2),
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

// TODO: Implement `FromGeometry` for points and vectors with `FloatProxy`
//       scalars once specialization lands. This isn't possible now, because it
//       would conflict with the blanket implementation for some shared scalar
//       type `T`, which is arguably a more important implementation.
#[cfg(feature = "geometry-cgmath")]
mod feature_geometry_cgmath {
    use cgmath::{BaseFloat, BaseNum, Point2, Point3, Vector2, Vector3};
    use decorum::{Finite, NotNan, Ordered, Primitive};

    use geometry::{Duplet, Triplet};
    use geometry::convert::*;

    // TODO: Implement `FromGeometry` for proxy types via specialization.
    // TODO: Implement these conversions for two-dimensional points.
    macro_rules! ordered {
        (geometry => $g:ident, proxy => $p:ident) => {
            impl<T> FromGeometry<$g<$p<T>>> for $g<T>
            where
                T: BaseFloat + BaseNum + Primitive,
            {
                fn from_geometry(other: $g<$p<T>>) -> Self {
                    $g::new(
                        other.x.into_inner(),
                        other.y.into_inner(),
                        other.z.into_inner(),
                    )
                }
            }

            impl<T> FromGeometry<$g<T>> for $g<$p<T>>
            where
                T: BaseFloat + BaseNum + Primitive,
            {
                fn from_geometry(other: $g<T>) -> Self {
                    $g::new(
                        $p::<T>::from_inner(other.x),
                        $p::<T>::from_inner(other.y),
                        $p::<T>::from_inner(other.z),
                    )
                }
            }
        };
    }
    ordered!(geometry => Point3, proxy => Finite);
    ordered!(geometry => Point3, proxy => NotNan);
    ordered!(geometry => Point3, proxy => Ordered);

    impl<T, U> FromGeometry<(U, U)> for Point2<T>
    where
        T: BaseNum + NumCast,
        U: ToPrimitive,
    {
        fn from_geometry(other: (U, U)) -> Self {
            Point2::new(T::from(other.0).unwrap(), T::from(other.1).unwrap())
        }
    }

    impl<T, U> FromGeometry<(U, U, U)> for Point3<T>
    where
        T: BaseNum + NumCast,
        U: ToPrimitive,
    {
        fn from_geometry(other: (U, U, U)) -> Self {
            Point3::new(
                T::from(other.0).unwrap(),
                T::from(other.1).unwrap(),
                T::from(other.2).unwrap(),
            )
        }
    }

    impl<T, U> FromGeometry<Duplet<U>> for Point2<T>
    where
        T: BaseNum + NumCast,
        U: ToPrimitive,
    {
        fn from_geometry(other: Duplet<U>) -> Self {
            Point2::new(T::from(other.0).unwrap(), T::from(other.1).unwrap())
        }
    }

    impl<T, U> FromGeometry<Triplet<U>> for Point3<T>
    where
        T: BaseNum + NumCast,
        U: ToPrimitive,
    {
        fn from_geometry(other: Triplet<U>) -> Self {
            Point3::new(
                T::from(other.0).unwrap(),
                T::from(other.1).unwrap(),
                T::from(other.2).unwrap(),
            )
        }
    }

    impl<T, U> FromGeometry<(U, U)> for Vector2<T>
    where
        T: BaseNum + NumCast,
        U: ToPrimitive,
    {
        fn from_geometry(other: (U, U)) -> Self {
            Vector2::new(T::from(other.0).unwrap(), T::from(other.1).unwrap())
        }
    }

    impl<T, U> FromGeometry<(U, U, U)> for Vector3<T>
    where
        T: BaseNum + NumCast,
        U: ToPrimitive,
    {
        fn from_geometry(other: (U, U, U)) -> Self {
            Vector3::new(
                T::from(other.0).unwrap(),
                T::from(other.1).unwrap(),
                T::from(other.2).unwrap(),
            )
        }
    }

    impl<T, U> FromGeometry<Duplet<U>> for Vector2<T>
    where
        T: BaseNum + NumCast,
        U: ToPrimitive,
    {
        fn from_geometry(other: Duplet<U>) -> Self {
            Vector2::new(T::from(other.0).unwrap(), T::from(other.1).unwrap())
        }
    }

    impl<T, U> FromGeometry<Triplet<U>> for Vector3<T>
    where
        T: BaseNum + NumCast,
        U: ToPrimitive,
    {
        fn from_geometry(other: Triplet<U>) -> Self {
            Vector3::new(
                T::from(other.0).unwrap(),
                T::from(other.1).unwrap(),
                T::from(other.2).unwrap(),
            )
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
            Point2::new(Ordered::from_inner(self.x), Ordered::from_inner(self.y))
        }

        fn from_hash(hash: Self::Hash) -> Self {
            Point2::new(hash.x.into_inner(), hash.y.into_inner())
        }
    }

    impl<T> HashConjugate for Point3<T>
    where
        T: BaseFloat + Float + Primitive,
    {
        type Hash = Point3<Ordered<T>>;

        fn into_hash(self) -> Self::Hash {
            self.into_geometry()
        }

        fn from_hash(hash: Self::Hash) -> Self {
            hash.into_geometry()
        }
    }

    impl<T> HashConjugate for Vector2<T>
    where
        T: BaseFloat + Float + Primitive,
    {
        type Hash = Vector2<Ordered<T>>;

        fn into_hash(self) -> Self::Hash {
            Vector2::new(Ordered::from_inner(self.x), Ordered::from_inner(self.y))
        }

        fn from_hash(hash: Self::Hash) -> Self {
            Vector2::new(hash.x.into_inner(), hash.y.into_inner())
        }
    }

    impl<T> HashConjugate for Vector3<T>
    where
        T: BaseFloat + Float + Primitive,
    {
        type Hash = Vector3<Ordered<T>>;

        fn into_hash(self) -> Self::Hash {
            Vector3::new(
                Ordered::from_inner(self.x),
                Ordered::from_inner(self.y),
                Ordered::from_inner(self.z),
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

#[cfg(feature = "geometry-nalgebra")]
mod feature_geometry_nalgebra {
    use decorum::{Finite, NotNan, Ordered, Primitive};
    use nalgebra::{Point2, Point3, Scalar, Vector2, Vector3};
    use num::{Float, NumCast, ToPrimitive};

    use geometry::{Duplet, Triplet};
    use geometry::convert::*;

    // TODO: Implement `FromGeometry` for proxy types via specialization.
    // TODO: Implement these conversions for two-dimensional points.
    macro_rules! ordered {
        (geometry => $g:ident, proxy => $p:ident) => {
            impl<T> FromGeometry<$g<$p<T>>> for $g<T>
            where
                T: Float + Primitive + Scalar,
            {
                fn from_geometry(other: $g<$p<T>>) -> Self {
                    $g::new(
                        other.x.into_inner(),
                        other.y.into_inner(),
                        other.z.into_inner(),
                    )
                }
            }

            impl<T> FromGeometry<$g<T>> for $g<$p<T>>
            where
                T: Float + Primitive + Scalar,
            {
                fn from_geometry(other: $g<T>) -> Self {
                    $g::new(
                        $p::<T>::from_inner(other.x),
                        $p::<T>::from_inner(other.y),
                        $p::<T>::from_inner(other.z),
                    )
                }
            }
        };
    }
    ordered!(geometry => Point3, proxy => Finite);
    ordered!(geometry => Point3, proxy => NotNan);
    ordered!(geometry => Point3, proxy => Ordered);

    impl<T, U> FromGeometry<(U, U)> for Point2<T>
    where
        T: NumCast + Scalar,
        U: ToPrimitive,
    {
        fn from_geometry(other: (U, U)) -> Self {
            Point2::new(T::from(other.0).unwrap(), T::from(other.1).unwrap())
        }
    }

    impl<T, U> FromGeometry<(U, U, U)> for Point3<T>
    where
        T: NumCast + Scalar,
        U: ToPrimitive,
    {
        fn from_geometry(other: (U, U, U)) -> Self {
            Point3::new(
                T::from(other.0).unwrap(),
                T::from(other.1).unwrap(),
                T::from(other.2).unwrap(),
            )
        }
    }

    impl<T, U> FromGeometry<Duplet<U>> for Point2<T>
    where
        T: NumCast + Scalar,
        U: ToPrimitive,
    {
        fn from_geometry(other: Duplet<U>) -> Self {
            Point2::new(T::from(other.0).unwrap(), T::from(other.1).unwrap())
        }
    }

    impl<T, U> FromGeometry<Triplet<U>> for Point3<T>
    where
        T: NumCast + Scalar,
        U: ToPrimitive,
    {
        fn from_geometry(other: Triplet<U>) -> Self {
            Point3::new(
                T::from(other.0).unwrap(),
                T::from(other.1).unwrap(),
                T::from(other.2).unwrap(),
            )
        }
    }

    impl<T, U> FromGeometry<(U, U)> for Vector2<T>
    where
        T: NumCast + Scalar,
        U: ToPrimitive,
    {
        fn from_geometry(other: (U, U)) -> Self {
            Vector2::new(T::from(other.0).unwrap(), T::from(other.1).unwrap())
        }
    }

    impl<T, U> FromGeometry<(U, U, U)> for Vector3<T>
    where
        T: NumCast + Scalar,
        U: ToPrimitive,
    {
        fn from_geometry(other: (U, U, U)) -> Self {
            Vector3::new(
                T::from(other.0).unwrap(),
                T::from(other.1).unwrap(),
                T::from(other.2).unwrap(),
            )
        }
    }

    impl<T, U> FromGeometry<Duplet<U>> for Vector2<T>
    where
        T: NumCast + Scalar,
        U: ToPrimitive,
    {
        fn from_geometry(other: Duplet<U>) -> Self {
            Vector2::new(T::from(other.0).unwrap(), T::from(other.1).unwrap())
        }
    }

    impl<T, U> FromGeometry<Triplet<U>> for Vector3<T>
    where
        T: NumCast + Scalar,
        U: ToPrimitive,
    {
        fn from_geometry(other: Triplet<U>) -> Self {
            Vector3::new(
                T::from(other.0).unwrap(),
                T::from(other.1).unwrap(),
                T::from(other.2).unwrap(),
            )
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
            Point2::new(Ordered::from_inner(self.x), Ordered::from_inner(self.y))
        }

        fn from_hash(hash: Self::Hash) -> Self {
            Point2::new(hash.x.into_inner(), hash.y.into_inner())
        }
    }

    impl<T> HashConjugate for Point3<T>
    where
        T: Float + Primitive + Scalar,
    {
        type Hash = Point3<Ordered<T>>;

        fn into_hash(self) -> Self::Hash {
            self.into_geometry()
        }

        fn from_hash(hash: Self::Hash) -> Self {
            hash.into_geometry()
        }
    }

    impl<T> HashConjugate for Vector2<T>
    where
        T: Float + Primitive + Scalar,
    {
        type Hash = Vector2<Ordered<T>>;

        fn into_hash(self) -> Self::Hash {
            Vector2::new(Ordered::from_inner(self.x), Ordered::from_inner(self.y))
        }

        fn from_hash(hash: Self::Hash) -> Self {
            Vector2::new(hash.x.into_inner(), hash.y.into_inner())
        }
    }

    impl<T> HashConjugate for Vector3<T>
    where
        T: Float + Primitive + Scalar,
    {
        type Hash = Vector3<Ordered<T>>;

        fn into_hash(self) -> Self::Hash {
            Vector3::new(
                Ordered::from_inner(self.x),
                Ordered::from_inner(self.y),
                Ordered::from_inner(self.z),
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
