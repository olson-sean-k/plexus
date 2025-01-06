#![cfg(feature = "geometry-cgmath")]

use cgmath::{Point2, Point3, Vector2, Vector3};
use decorum::{ExtendedReal, Primitive, Real, Total};
use num::{NumCast, ToPrimitive};

use crate::geometry::{FromGeometry, UnitGeometry};
use crate::graph::GraphData;

impl<T, U> FromGeometry<(U, U)> for Vector2<T>
where
    T: NumCast,
    U: ToPrimitive,
{
    fn from_geometry(other: (U, U)) -> Self {
        Vector2::new(T::from(other.0).unwrap(), T::from(other.1).unwrap())
    }
}

impl<T, U> FromGeometry<Vector2<T>> for (U, U)
where
    T: ToPrimitive,
    U: NumCast,
{
    fn from_geometry(other: Vector2<T>) -> Self {
        (U::from(other.x).unwrap(), U::from(other.y).unwrap())
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

impl<T, U> FromGeometry<Vector3<T>> for (U, U, U)
where
    T: ToPrimitive,
    U: NumCast,
{
    fn from_geometry(other: Vector3<T>) -> Self {
        (
            U::from(other.x).unwrap(),
            U::from(other.y).unwrap(),
            U::from(other.z).unwrap(),
        )
    }
}

impl<T, U> FromGeometry<(U, U)> for Point2<T>
where
    T: NumCast,
    U: ToPrimitive,
{
    fn from_geometry(other: (U, U)) -> Self {
        Point2::new(T::from(other.0).unwrap(), T::from(other.1).unwrap())
    }
}

impl<T, U> FromGeometry<Point2<T>> for (U, U)
where
    T: ToPrimitive,
    U: NumCast,
{
    fn from_geometry(other: Point2<T>) -> Self {
        (U::from(other.x).unwrap(), U::from(other.y).unwrap())
    }
}

impl<T> UnitGeometry for Point2<T> {}

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

impl<T, U> FromGeometry<Point3<T>> for (U, U, U)
where
    T: ToPrimitive,
    U: NumCast,
{
    fn from_geometry(other: Point3<T>) -> Self {
        (
            U::from(other.x).unwrap(),
            U::from(other.y).unwrap(),
            U::from(other.z).unwrap(),
        )
    }
}

impl<T> GraphData for Point2<T>
where
    Self: Copy,
{
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl<T> GraphData for Point3<T>
where
    Self: Copy,
{
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl<T> UnitGeometry for Point3<T> {}

macro_rules! with_constrained_scalars {
    ($f:ident) => {
        $f!(proxy => Real);
        $f!(proxy => ExtendedReal);
        $f!(proxy => Total);
    };
}

macro_rules! with_geometric_structures {
    ($f:ident) => {
        $f!(geometry => Vector2);
        $f!(geometry => Vector3);
        $f!(geometry => Point2);
        $f!(geometry => Point3);
    };
}

macro_rules! impl_from_geometry_for_constrained_scalar_structures {
    () => {
        with_constrained_scalars!(impl_from_geometry_for_constrained_scalar_structures);
    };
    (proxy => $p:ident) => {
        macro_rules! impl_from_geometry_for_scalar_structure {
            () => {
                with_geometric_structures!(impl_from_geometry_for_scalar_structure);
            };
            (geometry => $g:ident) => {
                impl<T> FromGeometry<$g<$p<T>>> for $g<T>
                where
                    T: Primitive,
                {
                    fn from_geometry(other: $g<$p<T>>) -> Self {
                        other.map(|value| value.into_inner())
                    }
                }

                impl<T> FromGeometry<$g<T>> for $g<$p<T>>
                where
                    T: Primitive,
                {
                    fn from_geometry(other: $g<T>) -> Self {
                        other.map($p::<T>::assert)
                    }
                }
            };
        }
        impl_from_geometry_for_scalar_structure!();
    };
}
impl_from_geometry_for_constrained_scalar_structures!();
