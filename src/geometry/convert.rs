use num::{NumCast, ToPrimitive};

use crate::geometry::{Duplet, Triplet};

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

/// Exposes a reference to positional vertex data.
///
/// To enable geometric features, this trait must be implemented for the type
/// representing vertex data. Additionally, geometric operations should be
/// implemented for the `Target` type.
pub trait AsPosition {
    type Target;

    fn as_position(&self) -> &Self::Target;
    fn as_position_mut(&mut self) -> &mut Self::Target;
}

// TODO: Implement `FromGeometry` for points and vectors with `FloatProxy`
//       scalars once specialization lands. This isn't possible now, because it
//       would conflict with the blanket implementation for some shared scalar
//       type `T`, which is arguably a more important implementation.
#[cfg(feature = "geometry-cgmath")]
mod feature_geometry_cgmath {
    use cgmath::{Point2, Point3, Vector2, Vector3};
    use decorum::{Finite, NotNan, Ordered, Primitive};
    use num::Float;

    use crate::geometry::convert::*;
    use crate::geometry::{Duplet, Triplet};

    // TODO: Implement `FromGeometry` for proxy types via specialization.
    // TODO: Implement these conversions for two-dimensional points.
    macro_rules! ordered {
        (geometry => $g:ident,proxy => $p:ident) => {
            impl<T> FromGeometry<$g<$p<T>>> for $g<T>
            where
                T: Float + Primitive,
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
                T: Float + Primitive,
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
        T: NumCast,
        U: ToPrimitive,
    {
        fn from_geometry(other: (U, U)) -> Self {
            Point2::new(T::from(other.0).unwrap(), T::from(other.1).unwrap())
        }
    }

    impl<T, U> FromGeometry<(U, U, U)> for Point3<T>
    where
        T: NumCast,
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
        T: NumCast,
        U: ToPrimitive,
    {
        fn from_geometry(other: Duplet<U>) -> Self {
            Point2::new(T::from(other.0).unwrap(), T::from(other.1).unwrap())
        }
    }

    impl<T, U> FromGeometry<Triplet<U>> for Point3<T>
    where
        T: NumCast,
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
        T: NumCast,
        U: ToPrimitive,
    {
        fn from_geometry(other: (U, U)) -> Self {
            Vector2::new(T::from(other.0).unwrap(), T::from(other.1).unwrap())
        }
    }

    impl<T, U> FromGeometry<(U, U, U)> for Vector3<T>
    where
        T: NumCast,
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
        T: NumCast,
        U: ToPrimitive,
    {
        fn from_geometry(other: Duplet<U>) -> Self {
            Vector2::new(T::from(other.0).unwrap(), T::from(other.1).unwrap())
        }
    }

    impl<T, U> FromGeometry<Triplet<U>> for Vector3<T>
    where
        T: NumCast,
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

    impl<T> AsPosition for Point2<T> {
        type Target = Self;

        fn as_position(&self) -> &Self::Target {
            self
        }

        fn as_position_mut(&mut self) -> &mut Self::Target {
            self
        }
    }

    impl<T> AsPosition for Point3<T> {
        type Target = Self;

        fn as_position(&self) -> &Self::Target {
            self
        }

        fn as_position_mut(&mut self) -> &mut Self::Target {
            self
        }
    }
}

#[cfg(feature = "geometry-mint")]
mod feature_geometry_mint {
    use decorum::{Finite, NotNan, Ordered, Primitive};
    use mint::{Point2, Point3, Vector2, Vector3};
    use num::{Float, NumCast, ToPrimitive};

    use crate::geometry::convert::*;
    use crate::geometry::{Duplet, Triplet};

    // TODO: Implement `FromGeometry` for proxy types via specialization.
    // TODO: Implement these conversions for two-dimensional points.
    macro_rules! ordered {
        (geometry => $g:ident,proxy => $p:ident) => {
            impl<T> FromGeometry<$g<$p<T>>> for $g<T>
            where
                T: Float + Primitive,
            {
                fn from_geometry(other: $g<$p<T>>) -> Self {
                    $g {
                        x: other.x.into_inner(),
                        y: other.y.into_inner(),
                        z: other.z.into_inner(),
                    }
                }
            }

            impl<T> FromGeometry<$g<T>> for $g<$p<T>>
            where
                T: Float + Primitive,
            {
                fn from_geometry(other: $g<T>) -> Self {
                    $g {
                        x: $p::<T>::from_inner(other.x),
                        y: $p::<T>::from_inner(other.y),
                        z: $p::<T>::from_inner(other.z),
                    }
                }
            }
        };
    }
    ordered!(geometry => Point3, proxy => Finite);
    ordered!(geometry => Point3, proxy => NotNan);
    ordered!(geometry => Point3, proxy => Ordered);

    impl<T, U> FromGeometry<(U, U)> for Point2<T>
    where
        T: NumCast,
        U: ToPrimitive,
    {
        fn from_geometry(other: (U, U)) -> Self {
            Point2 {
                x: T::from(other.0).unwrap(),
                y: T::from(other.1).unwrap(),
            }
        }
    }

    impl<T, U> FromGeometry<(U, U, U)> for Point3<T>
    where
        T: NumCast,
        U: ToPrimitive,
    {
        fn from_geometry(other: (U, U, U)) -> Self {
            Point3 {
                x: T::from(other.0).unwrap(),
                y: T::from(other.1).unwrap(),
                z: T::from(other.2).unwrap(),
            }
        }
    }

    impl<T, U> FromGeometry<Duplet<U>> for Point2<T>
    where
        T: NumCast,
        U: ToPrimitive,
    {
        fn from_geometry(other: Duplet<U>) -> Self {
            Point2 {
                x: T::from(other.0).unwrap(),
                y: T::from(other.1).unwrap(),
            }
        }
    }

    impl<T, U> FromGeometry<Triplet<U>> for Point3<T>
    where
        T: NumCast,
        U: ToPrimitive,
    {
        fn from_geometry(other: Triplet<U>) -> Self {
            Point3 {
                x: T::from(other.0).unwrap(),
                y: T::from(other.1).unwrap(),
                z: T::from(other.2).unwrap(),
            }
        }
    }

    impl<T, U> FromGeometry<(U, U)> for Vector2<T>
    where
        T: NumCast,
        U: ToPrimitive,
    {
        fn from_geometry(other: (U, U)) -> Self {
            Vector2 {
                x: T::from(other.0).unwrap(),
                y: T::from(other.1).unwrap(),
            }
        }
    }

    impl<T, U> FromGeometry<(U, U, U)> for Vector3<T>
    where
        T: NumCast,
        U: ToPrimitive,
    {
        fn from_geometry(other: (U, U, U)) -> Self {
            Vector3 {
                x: T::from(other.0).unwrap(),
                y: T::from(other.1).unwrap(),
                z: T::from(other.2).unwrap(),
            }
        }
    }

    impl<T, U> FromGeometry<Duplet<U>> for Vector2<T>
    where
        T: NumCast,
        U: ToPrimitive,
    {
        fn from_geometry(other: Duplet<U>) -> Self {
            Vector2 {
                x: T::from(other.0).unwrap(),
                y: T::from(other.1).unwrap(),
            }
        }
    }

    impl<T, U> FromGeometry<Triplet<U>> for Vector3<T>
    where
        T: NumCast,
        U: ToPrimitive,
    {
        fn from_geometry(other: Triplet<U>) -> Self {
            Vector3 {
                x: T::from(other.0).unwrap(),
                y: T::from(other.1).unwrap(),
                z: T::from(other.2).unwrap(),
            }
        }
    }

    impl<T> AsPosition for Point2<T> {
        type Target = Self;

        fn as_position(&self) -> &Self::Target {
            self
        }

        fn as_position_mut(&mut self) -> &mut Self::Target {
            self
        }
    }

    impl<T> AsPosition for Point3<T> {
        type Target = Self;

        fn as_position(&self) -> &Self::Target {
            self
        }

        fn as_position_mut(&mut self) -> &mut Self::Target {
            self
        }
    }
}

#[cfg(feature = "geometry-nalgebra")]
mod feature_geometry_nalgebra {
    use decorum::{Finite, NotNan, Ordered, Primitive};
    use nalgebra::{Point2, Point3, Scalar, Vector2, Vector3};
    use num::{Float, NumCast, ToPrimitive};

    use crate::geometry::convert::*;
    use crate::geometry::{Duplet, Triplet};

    // TODO: Implement `FromGeometry` for proxy types via specialization.
    // TODO: Implement these conversions for two-dimensional points.
    macro_rules! ordered {
        (geometry => $g:ident,proxy => $p:ident) => {
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

    impl<T> AsPosition for Point2<T>
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
}
