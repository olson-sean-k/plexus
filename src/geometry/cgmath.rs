#![cfg(feature = "geometry-cgmath")]

use cgmath::{self, Point2, Point3, Vector2, Vector3};
use decorum::{Finite, NotNan, Ordered, Primitive};
use num::{Float, NumCast, ToPrimitive};

use crate::geometry::{AsPosition, FromGeometry, Geometry};

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

macro_rules! from_ordered_geometry {
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
                other.map(|value| $p::<T>::from_inner(value))
            }
        }
    };
}
from_ordered_geometry!(geometry => Point2, proxy => Finite);
from_ordered_geometry!(geometry => Point2, proxy => NotNan);
from_ordered_geometry!(geometry => Point2, proxy => Ordered);
from_ordered_geometry!(geometry => Point3, proxy => Finite);
from_ordered_geometry!(geometry => Point3, proxy => NotNan);
from_ordered_geometry!(geometry => Point3, proxy => Ordered);

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

impl<T> Geometry for Point2<T>
where
    T: Clone,
{
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl<T> Geometry for Point3<T>
where
    T: Clone,
{
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}
