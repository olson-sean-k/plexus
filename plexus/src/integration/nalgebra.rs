#![cfg(feature = "geometry-nalgebra")]

use decorum::{ExtendedReal, Primitive, Real, Total};
use nalgebra::base::allocator::Allocator;
use nalgebra::{
    DefaultAllocator, DimName, OMatrix, OPoint, Point2, Point3, Scalar, Vector2, Vector3,
};
use num::{NumCast, ToPrimitive};

use crate::geometry::{FromGeometry, UnitGeometry};
use crate::graph::GraphData;

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

impl<T, D> GraphData for OPoint<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<D>,
    Self: Copy,
{
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl<T, D> UnitGeometry for OPoint<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<D>,
{
}

macro_rules! with_constrained_scalars {
    ($f:ident) => {
        $f!(proxy => Real);
        $f!(proxy => ExtendedReal);
        $f!(proxy => Total);
    };
}

macro_rules! impl_from_geometry_for_constrained_scalar_structures {
    () => {
        with_constrained_scalars!(impl_from_geometry_for_constrained_scalar_structures);
    };
    (proxy => $p:ident) => {
        impl<T, R, C> FromGeometry<OMatrix<$p<T>, R, C>> for OMatrix<T, R, C>
        where
            T: Primitive + Scalar,
            R: DimName,
            C: DimName,
            DefaultAllocator: Allocator<R, C>,
        {
            fn from_geometry(other: OMatrix<$p<T>, R, C>) -> Self {
                other.map(|value| value.into_inner())
            }
        }

        impl<T, R, C> FromGeometry<OMatrix<T, R, C>> for OMatrix<$p<T>, R, C>
        where
            T: Primitive + Scalar,
            R: DimName,
            C: DimName,
            DefaultAllocator: Allocator<R, C>,
        {
            fn from_geometry(other: OMatrix<T, R, C>) -> Self {
                other.map($p::<T>::assert)
            }
        }

        impl<T, D> FromGeometry<OPoint<$p<T>, D>> for OPoint<T, D>
        where
            T: Primitive + Scalar,
            D: DimName,
            DefaultAllocator: Allocator<D>,
        {
            fn from_geometry(other: OPoint<$p<T>, D>) -> Self {
                OPoint::from(other.coords.map(|value| value.into_inner()))
            }
        }

        impl<T, D> FromGeometry<OPoint<T, D>> for OPoint<$p<T>, D>
        where
            T: Primitive + Scalar,
            D: DimName,
            DefaultAllocator: Allocator<D>,
        {
            fn from_geometry(other: OPoint<T, D>) -> Self {
                OPoint::from(other.coords.map($p::<T>::assert))
            }
        }
    };
}
impl_from_geometry_for_constrained_scalar_structures!();
