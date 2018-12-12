use decorum::{Real, R64};
use num::{Num, NumCast};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use crate::geometry::{self, Duplet, Triplet};
use crate::Half;

pub trait Normalize {
    fn normalize(self) -> Self;
}

pub trait Average: Sized {
    fn average<I>(values: I) -> Self
    where
        I: IntoIterator<Item = Self>;
}

pub trait Interpolate<T = Self>: Sized {
    type Output;

    fn lerp(self, other: T, f: R64) -> Self::Output;

    fn midpoint(self, other: T) -> Self::Output {
        self.lerp(other, Half::half())
    }
}

pub trait Dot<T = Self> {
    type Output;

    fn dot(self, other: T) -> Self::Output;
}

pub trait Cross<T = Self> {
    type Output;

    fn cross(self, other: T) -> Self::Output;
}

pub trait Project<T = Self> {
    type Output;

    fn project(self, other: T) -> Self::Output;
}

impl<T> Project<T> for T
where
    T: Dot + Clone + Mul<<<Self as Dot>::Output as Div>::Output, Output = Self>,
    <T as Dot>::Output: Div,
{
    type Output = T;

    fn project(self, other: T) -> Self::Output {
        let n = other.dot(self.clone());
        let d = self.clone().dot(self.clone());
        self * (n / d)
    }
}

impl<T> Interpolate for Duplet<T>
where
    T: Copy + Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        Duplet(
            geometry::lerp(self.0, other.0, f),
            geometry::lerp(self.1, other.1, f),
        )
    }
}

impl<T> Normalize for Duplet<T>
where
    T: Real,
{
    fn normalize(self) -> Self {
        let m = (self.0.powi(2) + self.1.powi(2)).sqrt();
        Duplet(self.0 / m, self.1 / m)
    }
}

impl<T> Average for Duplet<T>
where
    T: AddAssign + Clone + Num + NumCast,
{
    fn average<I>(values: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        let (n, sum) = {
            let mut n = T::zero();
            let mut sum = Duplet(T::zero(), T::zero());
            for point in values {
                n += T::one();
                sum = Duplet(sum.0 + point.0, sum.1 + point.1);
            }
            (n, sum)
        };
        let m = T::one() / n;
        Duplet(sum.0 * m.clone(), sum.1 * m)
    }
}

impl<T> Dot for Duplet<T>
where
    T: Mul,
    <T as Mul>::Output: Add<Output = T>,
{
    type Output = T;

    fn dot(self, other: Self) -> Self::Output {
        (self.0 * other.0) + (self.1 * other.1)
    }
}

impl<T> Interpolate for Triplet<T>
where
    T: Copy + Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        Triplet(
            geometry::lerp(self.0, other.0, f),
            geometry::lerp(self.1, other.1, f),
            geometry::lerp(self.2, other.2, f),
        )
    }
}

impl<T> Normalize for Triplet<T>
where
    T: Real,
{
    fn normalize(self) -> Self {
        let m = (self.0.powi(2) + self.1.powi(2) + self.2.powi(2)).sqrt();
        Triplet(self.0 / m, self.1 / m, self.2 / m)
    }
}

impl<T> Average for Triplet<T>
where
    T: AddAssign + Clone + Num + NumCast,
{
    fn average<I>(values: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        let (n, sum) = {
            let mut n = T::zero();
            let mut sum = Triplet(T::zero(), T::zero(), T::zero());
            for point in values {
                n += T::one();
                sum = Triplet(sum.0 + point.0, sum.1 + point.1, sum.2 + point.2);
            }
            (n, sum)
        };
        let m = T::one() / n;
        Triplet(sum.0 * m.clone(), sum.1 * m.clone(), sum.2 * m)
    }
}

impl<T> Dot for Triplet<T>
where
    T: Mul<Output = T>,
    <T as Mul>::Output: Add<Output = T>,
{
    type Output = T;

    fn dot(self, other: Self) -> Self::Output {
        (self.0 * other.0) + (self.1 * other.1) + (self.2 * other.2)
    }
}

impl<T> Cross for Triplet<T>
where
    T: Clone + Mul + Neg,
    <T as Mul>::Output: Sub<Output = T>,
    <<T as Mul>::Output as Sub>::Output: Neg<Output = T>,
{
    type Output = Self;

    fn cross(self, other: Self) -> Self::Output {
        Triplet(
            (self.1.clone() * other.2.clone()) - (self.2.clone() * other.1.clone()),
            -((self.0.clone() * other.2.clone()) - (self.2 * other.0.clone())),
            (self.0.clone() * other.1.clone()) - (self.1 * other.0),
        )
    }
}

#[cfg(feature = "geometry-cgmath")]
mod feature_geometry_cgmath {
    use cgmath::{
        BaseFloat, BaseNum, EuclideanSpace, InnerSpace, Point2, Point3, Vector2, Vector3,
    };
    use num::{Num, NumCast};
    use std::ops::AddAssign;

    use crate::geometry;
    use crate::geometry::ops::*;

    impl<T> Normalize for Vector2<T>
    where
        T: BaseFloat,
    {
        fn normalize(self) -> Self {
            <Self as InnerSpace>::normalize(self)
        }
    }

    impl<T> Normalize for Vector3<T>
    where
        T: BaseFloat,
    {
        fn normalize(self) -> Self {
            <Self as InnerSpace>::normalize(self)
        }
    }

    impl<T> Average for Point2<T>
    where
        T: AddAssign + BaseNum + NumCast,
    {
        fn average<I>(values: I) -> Self
        where
            I: IntoIterator<Item = Self>,
        {
            let (n, sum) = {
                let mut n = T::zero();
                let mut sum = Point2::origin();
                for point in values {
                    n += T::one();
                    sum += Vector2::<T>::new(point.x, point.y);
                }
                (n, sum)
            };
            sum * (T::one() / n)
        }
    }

    impl<T> Average for Point3<T>
    where
        T: AddAssign + BaseNum + NumCast,
    {
        fn average<I>(values: I) -> Self
        where
            I: IntoIterator<Item = Self>,
        {
            let (n, sum) = {
                let mut n = T::zero();
                let mut sum = Point3::origin();
                for point in values {
                    n += T::one();
                    sum += Vector3::<T>::new(point.x, point.y, point.z);
                }
                (n, sum)
            };
            sum * (T::one() / n)
        }
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

    impl<T> Dot for Vector2<T>
    where
        T: BaseFloat,
    {
        type Output = T;

        fn dot(self, other: Self) -> Self::Output {
            <Self as InnerSpace>::dot(self, other)
        }
    }

    impl<T> Dot for Vector3<T>
    where
        T: BaseFloat,
    {
        type Output = T;

        fn dot(self, other: Self) -> Self::Output {
            <Self as InnerSpace>::dot(self, other)
        }
    }

    impl<T> Cross for Vector3<T>
    where
        T: BaseFloat,
    {
        type Output = Self;

        fn cross(self, other: Self) -> Self::Output {
            Self::cross(self, other)
        }
    }
}

#[cfg(feature = "geometry-mint")]
mod feature_geometry_mint {
    use decorum::Real;
    use mint::{Point2, Point3, Vector2, Vector3};
    use num::{Num, NumCast};
    use std::ops::{Add, Mul, Neg, Sub};

    use crate::geometry;
    use crate::geometry::ops::*;

    impl<T> Normalize for Vector2<T>
    where
        T: Real,
    {
        fn normalize(self) -> Self {
            let m = (self.x.powi(2) + self.y.powi(2)).sqrt();
            Vector2 {
                x: self.x / m,
                y: self.y / m,
            }
        }
    }

    impl<T> Normalize for Vector3<T>
    where
        T: Real,
    {
        fn normalize(self) -> Self {
            let m = (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt();
            Vector3 {
                x: self.x / m,
                y: self.y / m,
                z: self.z / m,
            }
        }
    }

    impl<T> Average for Point2<T>
    where
        T: AddAssign + Clone + Num + NumCast,
    {
        fn average<I>(values: I) -> Self
        where
            I: IntoIterator<Item = Self>,
        {
            let (n, sum) = {
                let mut n = T::zero();
                let mut sum = Point2 {
                    x: T::zero(),
                    y: T::zero(),
                };
                for point in values {
                    n += T::one();
                    sum = Point2 {
                        x: sum.x + point.x,
                        y: sum.y + point.y,
                    };
                }
                (n, sum)
            };
            let m = T::one() / n;
            Point2 {
                x: sum.x * m.clone(),
                y: sum.y * m,
            }
        }
    }

    impl<T> Average for Point3<T>
    where
        T: AddAssign + Clone + Num + NumCast,
    {
        fn average<I>(values: I) -> Self
        where
            I: IntoIterator<Item = Self>,
        {
            let (n, sum) = {
                let mut n = T::zero();
                let mut sum = Point3 {
                    x: T::zero(),
                    y: T::zero(),
                    z: T::zero(),
                };
                for point in values {
                    n += T::one();
                    sum = Point3 {
                        x: sum.x + point.x,
                        y: sum.y + point.y,
                        z: sum.z + point.z,
                    };
                }
                (n, sum)
            };
            let m = T::one() / n;
            Point3 {
                x: sum.x * m.clone(),
                y: sum.y * m.clone(),
                z: sum.z * m,
            }
        }
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

    impl<T> Dot for Vector2<T>
    where
        T: Mul,
        <T as Mul>::Output: Add<Output = T>,
    {
        type Output = T;

        fn dot(self, other: Self) -> Self::Output {
            (self.x * other.x) + (self.y * other.y)
        }
    }

    impl<T> Dot for Vector3<T>
    where
        T: Mul<Output = T>,
        <T as Mul>::Output: Add<Output = T>,
    {
        type Output = T;

        fn dot(self, other: Self) -> Self::Output {
            (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
        }
    }

    impl<T> Cross for Vector3<T>
    where
        T: Clone + Mul + Neg,
        <T as Mul>::Output: Sub<Output = T>,
        <<T as Mul>::Output as Sub>::Output: Neg<Output = T>,
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
}

#[cfg(feature = "geometry-nalgebra")]
mod feature_geometry_nalgebra {
    use decorum::Real;
    use nalgebra::core::Matrix;
    use nalgebra::{Point2, Point3, Scalar, Vector2, Vector3};
    use num::{Num, NumCast, Zero};
    use std::ops::{AddAssign, Mul, MulAssign, Neg, Sub};

    use crate::geometry;
    use crate::geometry::ops::*;

    impl<T> Normalize for Vector2<T>
    where
        T: Real + Scalar,
    {
        // nalgebra provides an implementation via:
        //
        // ```rust
        // Matrix::normalize(&self)
        // ```
        //
        // However, that requires a bound on nalgebra's `Real` trait, which is
        // only implemented for a limited set of types.
        fn normalize(self) -> Self {
            let m = (self.x.powi(2) + self.y.powi(2)).sqrt();
            Vector2::new(self.x / m, self.y / m)
        }
    }

    impl<T> Normalize for Vector3<T>
    where
        T: Real + Scalar,
    {
        // nalgebra provides an implementation via:
        //
        // ```rust
        // Matrix::normalize(&self)
        // ```
        //
        // However, that requires a bound on nalgebra's `Real` trait, which is
        // only implemented for a limited set of types.
        fn normalize(self) -> Self {
            let m = (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt();
            Vector3::new(self.x / m, self.y / m, self.z / m)
        }
    }

    // TODO: Implement `Average` for points and vectors of arbitrary dimension.
    impl<T> Average for Point2<T>
    where
        T: AddAssign + MulAssign + Num + NumCast + Scalar,
    {
        fn average<I>(values: I) -> Self
        where
            I: IntoIterator<Item = Self>,
        {
            let (n, sum) = {
                let mut n = T::zero();
                let mut sum = Point2::origin();
                for point in values {
                    n += T::one();
                    sum += Vector2::<T>::new(point.x, point.y);
                }
                (n, sum)
            };
            sum * (T::one() / n)
        }
    }

    impl<T> Average for Point3<T>
    where
        T: AddAssign + MulAssign + Num + NumCast + Scalar,
    {
        fn average<I>(values: I) -> Self
        where
            I: IntoIterator<Item = Self>,
        {
            let (n, sum) = {
                let mut n = T::zero();
                let mut sum = Point3::origin();
                for point in values {
                    n += T::one();
                    sum += Vector3::<T>::new(point.x, point.y, point.z);
                }
                (n, sum)
            };
            sum * (T::one() / n)
        }
    }

    impl<T> Interpolate for Point2<T>
    where
        T: Num + NumCast + Scalar,
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
        T: Num + NumCast + Scalar,
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

    impl<T> Interpolate for Vector2<T>
    where
        T: Num + NumCast + Scalar,
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
        T: Num + NumCast + Scalar,
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

    impl<T> Dot for Vector2<T>
    where
        T: AddAssign + Mul<Output = T> + MulAssign + Scalar + Zero,
    {
        type Output = T;

        fn dot(self, other: Self) -> Self::Output {
            Matrix::dot(&self, &other)
        }
    }

    impl<T> Dot for Vector3<T>
    where
        T: AddAssign + Mul<Output = T> + MulAssign + Scalar + Zero,
    {
        type Output = T;

        fn dot(self, other: Self) -> Self::Output {
            Matrix::dot(&self, &other)
        }
    }

    impl<T> Cross for Vector3<T>
    where
        T: Mul + Neg + Scalar,
        <T as Mul>::Output: Sub<Output = T>,
        <<T as Mul>::Output as Sub>::Output: Neg<Output = T>,
    {
        type Output = Self;

        // nalgebra provides an implementation via:
        //
        // ```rust
        // Matrix::cross(&self, &other)
        // ```
        //
        // However, that requires a bound on alga's `AbstractRing` trait.
        fn cross(self, other: Self) -> Self::Output {
            Vector3::new(
                (self.y * other.z) - (self.z * other.y),
                -((self.x * other.z) - (self.z * other.x)),
                (self.x * other.y) - (self.y * other.x),
            )
        }
    }
}
