#![cfg(feature = "geometry-cgmath")]

use cgmath::{self, BaseFloat, BaseNum, Point2, Point3, Vector2, Vector3};
use decorum::{Finite, NotNan, Ordered, Primitive, Real, R64};
use num::{Float, Num, NumCast, ToPrimitive};

use crate::geometry::convert::{AsPosition, FromGeometry};
use crate::geometry::ops::{Cross, Dot, Interpolate};
use crate::geometry::space::{EuclideanSpace, InnerSpace, VectorSpace};
use crate::geometry::{self, Duplet, Geometry, Triplet};

impl<T> Cross for Vector3<T>
where
    T: BaseFloat,
{
    type Output = Self;

    fn cross(self, other: Self) -> Self::Output {
        Self::cross(self, other)
    }
}

impl<T> Dot for Vector2<T>
where
    T: BaseFloat,
{
    type Output = T;

    fn dot(self, other: Self) -> Self::Output {
        <Self as cgmath::InnerSpace>::dot(self, other)
    }
}

impl<T> Dot for Vector3<T>
where
    T: BaseFloat,
{
    type Output = T;

    fn dot(self, other: Self) -> Self::Output {
        <Self as cgmath::InnerSpace>::dot(self, other)
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

impl<T> InnerSpace for Vector2<T> where T: BaseFloat + Real {}

impl<T> InnerSpace for Vector3<T> where T: BaseFloat + Real {}

impl<T> Interpolate for Vector2<T>
where
    T: Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        Vector2::new(
            geometry::lerp(self.x, other.x, f),
            geometry::lerp(self.y, other.y, f),
        )
    }
}

impl<T> Interpolate for Vector3<T>
where
    T: Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        Vector3::new(
            geometry::lerp(self.x, other.x, f),
            geometry::lerp(self.y, other.y, f),
            geometry::lerp(self.z, other.z, f),
        )
    }
}

impl<T> VectorSpace for Vector2<T>
where
    T: BaseNum + Real,
{
    type Scalar = T;
}

impl<T> VectorSpace for Vector3<T>
where
    T: BaseNum + Real,
{
    type Scalar = T;
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

impl<T> EuclideanSpace for Point2<T>
where
    T: BaseFloat + BaseNum + Real,
{
    type Difference = Vector2<T>;

    fn origin() -> Self {
        <Self as cgmath::EuclideanSpace>::origin()
    }
}

impl<T> EuclideanSpace for Point3<T>
where
    T: BaseFloat + BaseNum + Real,
{
    type Difference = Vector3<T>;

    fn origin() -> Self {
        <Self as cgmath::EuclideanSpace>::origin()
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

impl<T> Interpolate for Point2<T>
where
    T: Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        Point2::new(
            geometry::lerp(self.x, other.x, f),
            geometry::lerp(self.y, other.y, f),
        )
    }
}

impl<T> Interpolate for Point3<T>
where
    T: Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        Point3::new(
            geometry::lerp(self.x, other.x, f),
            geometry::lerp(self.y, other.y, f),
            geometry::lerp(self.z, other.z, f),
        )
    }
}

impl<T, U> From<Point2<U>> for Duplet<T>
where
    T: NumCast,
    U: ToPrimitive,
{
    fn from(other: Point2<U>) -> Self {
        Duplet(T::from(other.x).unwrap(), T::from(other.y).unwrap())
    }
}

impl<T, U> From<Vector2<U>> for Duplet<T>
where
    T: NumCast,
    U: ToPrimitive,
{
    fn from(other: Vector2<U>) -> Self {
        Duplet(T::from(other.x).unwrap(), T::from(other.y).unwrap())
    }
}

impl<T, U> Into<Point2<T>> for Duplet<U>
where
    T: NumCast,
    U: ToPrimitive,
{
    fn into(self) -> Point2<T> {
        Point2::new(T::from(self.0).unwrap(), T::from(self.1).unwrap())
    }
}

impl<T, U> Into<Vector2<T>> for Duplet<U>
where
    T: NumCast,
    U: ToPrimitive,
{
    fn into(self) -> Vector2<T> {
        Vector2::new(T::from(self.0).unwrap(), T::from(self.1).unwrap())
    }
}

impl<T, U> From<Point3<U>> for Triplet<T>
where
    T: NumCast,
    U: ToPrimitive,
{
    fn from(other: Point3<U>) -> Self {
        Triplet(
            T::from(other.x).unwrap(),
            T::from(other.y).unwrap(),
            T::from(other.z).unwrap(),
        )
    }
}

impl<T, U> From<Vector3<U>> for Triplet<T>
where
    T: NumCast,
    U: ToPrimitive,
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
    T: NumCast,
    U: ToPrimitive,
{
    fn into(self) -> Point3<T> {
        Point3::new(
            T::from(self.0).unwrap(),
            T::from(self.1).unwrap(),
            T::from(self.2).unwrap(),
        )
    }
}

impl<T, U> Into<Vector3<T>> for Triplet<U>
where
    T: NumCast,
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
