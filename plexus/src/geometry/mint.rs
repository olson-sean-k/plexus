#![cfg(feature = "geometry-mint")]

use decorum::{Finite, NotNan, Ordered, Primitive};
use mint::{Point2, Point3, Vector2, Vector3};
use num::{Float, NumCast, ToPrimitive};

use crate::geometry::{AsPosition, FromGeometry};
use crate::graph::GraphGeometry;

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
        Vector3 {
            x: T::from(other.0).unwrap(),
            y: T::from(other.1).unwrap(),
            z: T::from(other.2).unwrap(),
        }
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
    (proxy => $p:ident) => {
        impl<T> FromGeometry<Point2<$p<T>>> for Point2<T>
        where
            T: Float + Primitive,
        {
            fn from_geometry(other: Point2<$p<T>>) -> Self {
                Point2 {
                    x: other.x.into_inner(),
                    y: other.y.into_inner(),
                }
            }
        }

        impl<T> FromGeometry<Point2<T>> for Point2<$p<T>>
        where
            T: Float + Primitive,
        {
            fn from_geometry(other: Point2<T>) -> Self {
                Point2 {
                    x: $p::<T>::from_inner(other.x),
                    y: $p::<T>::from_inner(other.y),
                }
            }
        }

        impl<T> FromGeometry<Point3<$p<T>>> for Point3<T>
        where
            T: Float + Primitive,
        {
            fn from_geometry(other: Point3<$p<T>>) -> Self {
                Point3 {
                    x: other.x.into_inner(),
                    y: other.y.into_inner(),
                    z: other.z.into_inner(),
                }
            }
        }

        impl<T> FromGeometry<Point3<T>> for Point3<$p<T>>
        where
            T: Float + Primitive,
        {
            fn from_geometry(other: Point3<T>) -> Self {
                Point3 {
                    x: $p::<T>::from_inner(other.x),
                    y: $p::<T>::from_inner(other.y),
                    z: $p::<T>::from_inner(other.z),
                }
            }
        }
    };
}
from_ordered_geometry!(proxy => Finite);
from_ordered_geometry!(proxy => NotNan);
from_ordered_geometry!(proxy => Ordered);

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
        Point3 {
            x: T::from(other.0).unwrap(),
            y: T::from(other.1).unwrap(),
            z: T::from(other.2).unwrap(),
        }
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
