#![cfg(feature = "geometry-cgmath")]

use theon::integration::cgmath;

#[doc(hidden)]
pub use self::cgmath::*;

use decorum::{Finite, NotNan, Ordered, Primitive};
use num::{Float, NumCast, ToPrimitive};

use crate::graph::GraphGeometry;
use crate::{FromGeometry, UnitGeometry};

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

impl<T> GraphGeometry for Point2<T>
where
    Self: Copy,
{
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl<T> GraphGeometry for Point3<T>
where
    Self: Copy,
{
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl<T> UnitGeometry for Point3<T> {}

macro_rules! impl_from_geometry_ordered {
    (geometry => $g:ident,proxy => $p:ident) => {
        impl<T> FromGeometry<$g<$p<T>>> for $g<T>
        where
            T: Float + Primitive,
        {
            fn from_geometry(other: $g<$p<T>>) -> Self {
                other.map(|value| value.into_inner())
            }
        }

        impl<T> FromGeometry<$g<T>> for $g<$p<T>>
        where
            T: Float + Primitive,
        {
            fn from_geometry(other: $g<T>) -> Self {
                other.map($p::<T>::from_inner)
            }
        }
    };
}
impl_from_geometry_ordered!(geometry => Vector2, proxy => Finite);
impl_from_geometry_ordered!(geometry => Vector2, proxy => NotNan);
impl_from_geometry_ordered!(geometry => Vector2, proxy => Ordered);
impl_from_geometry_ordered!(geometry => Vector3, proxy => Finite);
impl_from_geometry_ordered!(geometry => Vector3, proxy => NotNan);
impl_from_geometry_ordered!(geometry => Vector3, proxy => Ordered);
impl_from_geometry_ordered!(geometry => Point2, proxy => Finite);
impl_from_geometry_ordered!(geometry => Point2, proxy => NotNan);
impl_from_geometry_ordered!(geometry => Point2, proxy => Ordered);
impl_from_geometry_ordered!(geometry => Point3, proxy => Finite);
impl_from_geometry_ordered!(geometry => Point3, proxy => NotNan);
impl_from_geometry_ordered!(geometry => Point3, proxy => Ordered);
