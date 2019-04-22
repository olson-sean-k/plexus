#![cfg(feature = "geometry-mint")]

// TODO: It is not possible to implement vector space traits for `mint` types,
//       because they require foreign traits on foreign types.
// TODO: It could be useful to define more machinery for `mint` types so that
//       implementing geometric traits becomes easier. For example, zipping and
//       mapping would reduce repetition.

use decorum::{Finite, NotNan, Ordered, Primitive, R64};
use mint::{Point2, Point3, Vector2, Vector3};
use num::{Float, Num, NumCast, ToPrimitive, Zero};
use std::ops::Neg;

use crate::geometry::convert::{AsPosition, FromGeometry};
use crate::geometry::ops::{Cross, Dot, Interpolate};
use crate::geometry::space::{AbstractSpace, Origin};
use crate::geometry::{self, Duplet, Geometry, Triplet};

impl<T> AbstractSpace for Vector2<T>
where
    T: Clone + Neg<Output = T> + Num + NumCast,
{
    type Scalar = T;
}

impl<T> AbstractSpace for Vector3<T>
where
    T: Clone + Neg<Output = T> + Num + NumCast,
{
    type Scalar = T;
}

impl<T> Cross for Vector3<T>
where
    T: Clone + Neg<Output = T> + Num,
{
    type Output = Self;

    fn cross(self, other: Self) -> Self::Output {
        Vector3 {
            x: (self.y.clone() * other.z.clone()) - (self.z.clone() * other.y.clone()),
            y: -((self.x.clone() * other.z.clone()) - (self.z * other.x.clone())),
            z: (self.x.clone() * other.y.clone()) - (self.y * other.x),
        }
    }
}

impl<T> Dot for Vector2<T>
where
    T: Num,
{
    type Output = T;

    fn dot(self, other: Self) -> Self::Output {
        (self.x * other.x) + (self.y * other.y)
    }
}

impl<T> Dot for Vector3<T>
where
    T: Num,
{
    type Output = T;

    fn dot(self, other: Self) -> Self::Output {
        (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
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

impl<T> Interpolate for Vector2<T>
where
    T: Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        Vector2 {
            x: geometry::lerp(self.x, other.x, f),
            y: geometry::lerp(self.y, other.y, f),
        }
    }
}

impl<T> Interpolate for Vector3<T>
where
    T: Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        Vector3 {
            x: geometry::lerp(self.x, other.x, f),
            y: geometry::lerp(self.y, other.y, f),
            z: geometry::lerp(self.z, other.z, f),
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
        Point2 {
            x: geometry::lerp(self.x, other.x, f),
            y: geometry::lerp(self.y, other.y, f),
        }
    }
}

impl<T> Interpolate for Point3<T>
where
    T: Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        Point3 {
            x: geometry::lerp(self.x, other.x, f),
            y: geometry::lerp(self.y, other.y, f),
            z: geometry::lerp(self.z, other.z, f),
        }
    }
}

impl<T> Origin for Point2<T>
where
    T: Zero,
{
    fn origin() -> Self {
        Point2 {
            x: Zero::zero(),
            y: Zero::zero(),
        }
    }
}

impl<T> Origin for Point3<T>
where
    T: Zero,
{
    fn origin() -> Self {
        Point3 {
            x: Zero::zero(),
            y: Zero::zero(),
            z: Zero::zero(),
        }
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
        Point2 {
            x: T::from(self.0).unwrap(),
            y: T::from(self.1).unwrap(),
        }
    }
}

impl<T, U> Into<Vector2<T>> for Duplet<U>
where
    T: NumCast,
    U: ToPrimitive,
{
    fn into(self) -> Vector2<T> {
        Vector2 {
            x: T::from(self.0).unwrap(),
            y: T::from(self.1).unwrap(),
        }
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
        Point3 {
            x: T::from(self.0).unwrap(),
            y: T::from(self.1).unwrap(),
            z: T::from(self.2).unwrap(),
        }
    }
}

impl<T, U> Into<Vector3<T>> for Triplet<U>
where
    T: NumCast,
    U: ToPrimitive,
{
    fn into(self) -> Vector3<T> {
        Vector3 {
            x: T::from(self.0).unwrap(),
            y: T::from(self.1).unwrap(),
            z: T::from(self.2).unwrap(),
        }
    }
}
