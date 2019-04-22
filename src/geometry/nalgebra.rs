#![cfg(feature = "geometry-nalgebra")]

use alga::general::Ring;
use decorum::{Finite, NotNan, Ordered, Primitive, Real, R64};
use nalgebra::base::allocator::Allocator;
use nalgebra::base::default_allocator::DefaultAllocator;
use nalgebra::base::dimension::DimName;
use nalgebra::core::Matrix;
use nalgebra::{Point, Point2, Point3, Scalar, Vector2, Vector3, VectorN};
use num::{Float, Num, NumCast, ToPrimitive};
use std::ops::{AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::geometry::convert::{AsPosition, FromGeometry, IntoGeometry};
use crate::geometry::ops::{Cross, Dot, Interpolate};
use crate::geometry::space::{EuclideanSpace, VectorSpace};
use crate::geometry::{self, Duplet, Geometry, Triplet};

impl<T, D> VectorSpace for VectorN<T, D>
where
    T: AddAssign + Neg<Output = T> + MulAssign + Num + NumCast + Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    type Scalar = T;
}

impl<T> Cross for Vector3<T>
where
    T: Num + Ring + Scalar,
    <<T as Mul>::Output as Sub>::Output: Neg<Output = T>,
{
    type Output = Self;

    fn cross(self, other: Self) -> Self::Output {
        Matrix::cross(&self, &other)
    }
}

impl<T, D> Dot for VectorN<T, D>
where
    T: AddAssign + MulAssign + Num + Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    type Output = T;

    fn dot(self, other: Self) -> Self::Output {
        Matrix::dot(&self, &other)
    }
}

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

impl<T, D> Interpolate for VectorN<T, D>
where
    T: Num + NumCast + Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        self.zip_map(&other, |a, b| geometry::lerp(a, b, f))
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

impl<T, D> EuclideanSpace for Point<T, D>
where
    T: AddAssign + MulAssign + Neg<Output = T> + Real + Scalar + SubAssign,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    type Difference = VectorN<T, D>;

    fn origin() -> Self {
        Point::<T, D>::origin()
    }

    fn coordinates(&self) -> Self::Difference {
        self.coords.clone()
    }

    fn centroid<I>(points: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self>,
    {
        VectorSpace::mean(points.into_iter().map(|point| point.coords))
            .map(|mean| Point::from(mean))
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

impl<T, D> Interpolate for Point<T, D>
where
    T: Num + NumCast + Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        Point::from(self.coords.lerp(other.coords, f))
    }
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
