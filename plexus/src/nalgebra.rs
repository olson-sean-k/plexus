#![cfg(feature = "geometry-nalgebra")]

use decorum::{Finite, NotNan, Ordered, Primitive};
use nalgebra::base::allocator::Allocator;
use nalgebra::base::default_allocator::DefaultAllocator;
use nalgebra::base::dimension::DimName;
use nalgebra::{Point, Point2, Point3, Scalar, Vector2, Vector3};
use num::{Float, NumCast, ToPrimitive};

use crate::graph::GraphGeometry;
use crate::FromGeometry;

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
        (U::from(other.x).unwrap(), U::from(other.y).unwrap())
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
        (
            U::from(other.x).unwrap(),
            U::from(other.y).unwrap(),
            U::from(other.z).unwrap(),
        )
    }
}

macro_rules! from_ordered_geometry {
    (proxy => $p:ident) => {
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
from_ordered_geometry!(proxy => Finite);
from_ordered_geometry!(proxy => NotNan);
from_ordered_geometry!(proxy => Ordered);

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
        (U::from(other.x).unwrap(), U::from(other.y).unwrap())
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
        (
            U::from(other.x).unwrap(),
            U::from(other.y).unwrap(),
            U::from(other.z).unwrap(),
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
