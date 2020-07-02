#![cfg(feature = "geometry-nalgebra")]

use theon::integration::nalgebra;

#[doc(hidden)]
pub use self::nalgebra::*;

use self::nalgebra::base::allocator::Allocator;
use self::nalgebra::base::default_allocator::DefaultAllocator;
use self::nalgebra::base::dimension::DimName;
use decorum::{Finite, Float, NotNan, Primitive, Total};
use num::{NumCast, ToPrimitive};

use crate::geometry::{FromGeometry, UnitGeometry};
use crate::graph::GraphGeometry;

impl<T, U> FromGeometry<(U, U)> for Vector2<T>
where
    T: NumCast + Scalar,
    U: ToPrimitive,
{
    fn from_geometry(other: (U, U)) -> Self {
        Vector2::new(T::from(other.0).unwrap(), T::from(other.1).unwrap())
    }
}

impl<T, U> FromGeometry<Vector2<T>> for (U, U)
where
    T: Scalar + ToPrimitive,
    U: NumCast,
{
    fn from_geometry(other: Vector2<T>) -> Self {
        let [x, y]: [T; 2] = other.into();
        (U::from(x).unwrap(), U::from(y).unwrap())
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

impl<T, U> FromGeometry<Vector3<T>> for (U, U, U)
where
    T: Scalar + ToPrimitive,
    U: NumCast,
{
    fn from_geometry(other: Vector3<T>) -> Self {
        let [x, y, z]: [T; 3] = other.into();
        (
            U::from(x).unwrap(),
            U::from(y).unwrap(),
            U::from(z).unwrap(),
        )
    }
}

impl<T, U> FromGeometry<(U, U)> for Point2<T>
where
    T: NumCast + Scalar,
    U: ToPrimitive,
{
    fn from_geometry(other: (U, U)) -> Self {
        Point2::new(T::from(other.0).unwrap(), T::from(other.1).unwrap())
    }
}

impl<T, U> FromGeometry<Point2<T>> for (U, U)
where
    T: Scalar + ToPrimitive,
    U: NumCast,
{
    fn from_geometry(other: Point2<T>) -> Self {
        let [x, y]: [T; 2] = other.coords.into();
        (U::from(x).unwrap(), U::from(y).unwrap())
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

impl<T, U> FromGeometry<Point3<T>> for (U, U, U)
where
    T: Scalar + ToPrimitive,
    U: NumCast,
{
    fn from_geometry(other: Point3<T>) -> Self {
        let [x, y, z]: [T; 3] = other.coords.into();
        (
            U::from(x).unwrap(),
            U::from(y).unwrap(),
            U::from(z).unwrap(),
        )
    }
}

impl<T, D> GraphGeometry for Point<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
    Self: Copy,
{
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl<T, D> UnitGeometry for Point<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
}

macro_rules! impl_from_geometry_ordered {
    (proxy => $p:ident) => {
        impl<T, R, C> FromGeometry<MatrixMN<$p<T>, R, C>> for MatrixMN<T, R, C>
        where
            T: Float + Primitive + Scalar,
            R: DimName,
            C: DimName,
            DefaultAllocator: Allocator<T, R, C> + Allocator<$p<T>, R, C>,
        {
            fn from_geometry(other: MatrixMN<$p<T>, R, C>) -> Self {
                other.map(|value| value.into_inner())
            }
        }

        impl<T, R, C> FromGeometry<MatrixMN<T, R, C>> for MatrixMN<$p<T>, R, C>
        where
            T: Float + Primitive + Scalar,
            R: DimName,
            C: DimName,
            DefaultAllocator: Allocator<$p<T>, R, C> + Allocator<T, R, C>,
        {
            fn from_geometry(other: MatrixMN<T, R, C>) -> Self {
                other.map($p::<T>::from_inner)
            }
        }

        impl<T, D> FromGeometry<Point<$p<T>, D>> for Point<T, D>
        where
            T: Float + Primitive + Scalar,
            D: DimName,
            DefaultAllocator: Allocator<T, D> + Allocator<$p<T>, D>,
        {
            fn from_geometry(other: Point<$p<T>, D>) -> Self {
                Point::from(other.coords.map(|value| value.into_inner()))
            }
        }

        impl<T, D> FromGeometry<Point<T, D>> for Point<$p<T>, D>
        where
            T: Float + Primitive + Scalar,
            D: DimName,
            DefaultAllocator: Allocator<$p<T>, D> + Allocator<T, D>,
        {
            fn from_geometry(other: Point<T, D>) -> Self {
                Point::from(other.coords.map($p::<T>::from_inner))
            }
        }
    };
}
impl_from_geometry_ordered!(proxy => Finite);
impl_from_geometry_ordered!(proxy => NotNan);
impl_from_geometry_ordered!(proxy => Total);
