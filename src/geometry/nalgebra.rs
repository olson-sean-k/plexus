#![cfg(feature = "geometry-nalgebra")]

use decorum::{Finite, NotNan, Ordered, Primitive};
use nalgebra::base::allocator::Allocator;
use nalgebra::base::default_allocator::DefaultAllocator;
use nalgebra::base::dimension::DimName;
use nalgebra::{Point, Point2, Point3, Scalar, Vector2, Vector3, VectorN};
use num::{Float, NumCast, ToPrimitive};

use crate::geometry::convert::{AsPosition, FromGeometry, IntoGeometry};
use crate::geometry::{Duplet, Geometry, Triplet};

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

impl<T, D> AsPosition for Point<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
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
                Point::from(other.coords.map(|value| $p::<T>::from_inner(value)))
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
        Point2::from(IntoGeometry::<VectorN<_, _>>::into_geometry(other))
    }
}

impl<T, U> FromGeometry<Triplet<U>> for Point3<T>
where
    T: NumCast + Scalar,
    U: ToPrimitive,
{
    fn from_geometry(other: Triplet<U>) -> Self {
        Point3::from(IntoGeometry::<VectorN<_, _>>::into_geometry(other))
    }
}

impl<T, D> Geometry for Point<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl<T, U> From<Point2<U>> for Duplet<T>
where
    T: NumCast,
    U: Scalar + ToPrimitive,
{
    fn from(other: Point2<U>) -> Self {
        Duplet::from(other.coords)
    }
}

impl<T, U> From<Vector2<U>> for Duplet<T>
where
    T: NumCast,
    U: Scalar + ToPrimitive,
{
    fn from(other: Vector2<U>) -> Self {
        Duplet(T::from(other.x).unwrap(), T::from(other.y).unwrap())
    }
}

impl<T, U> Into<Point2<T>> for Duplet<U>
where
    T: NumCast + Scalar,
    U: ToPrimitive,
{
    fn into(self) -> Point2<T> {
        Point2::from(Into::<VectorN<_, _>>::into(self))
    }
}

impl<T, U> Into<Vector2<T>> for Duplet<U>
where
    T: NumCast + Scalar,
    U: ToPrimitive,
{
    fn into(self) -> Vector2<T> {
        Vector2::new(T::from(self.0).unwrap(), T::from(self.1).unwrap())
    }
}

impl<T, U> From<Point3<U>> for Triplet<T>
where
    T: NumCast,
    U: Scalar + ToPrimitive,
{
    fn from(other: Point3<U>) -> Self {
        Triplet::from(other.coords)
    }
}

impl<T, U> From<Vector3<U>> for Triplet<T>
where
    T: NumCast,
    U: Scalar + ToPrimitive,
{
    fn from(other: Vector3<U>) -> Self {
        Triplet(
            T::from(other.x).unwrap(),
            T::from(other.y).unwrap(),
            T::from(other.z).unwrap(),
        )
    }
}

impl<T, U> Into<Point3<T>> for Triplet<U>
where
    T: NumCast + Scalar,
    U: ToPrimitive,
{
    fn into(self) -> Point3<T> {
        Point3::from(Into::<VectorN<_, _>>::into(self))
    }
}

impl<T, U> Into<Vector3<T>> for Triplet<U>
where
    T: NumCast + Scalar,
    U: ToPrimitive,
{
    fn into(self) -> Vector3<T> {
        Vector3::new(
            T::from(self.0).unwrap(),
            T::from(self.1).unwrap(),
            T::from(self.2).unwrap(),
        )
    }
}
