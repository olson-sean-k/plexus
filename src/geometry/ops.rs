use num::{Num, NumCast};

use geometry::{self, Duplet, Triplet};

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

    fn lerp(self, other: T, f: f64) -> Self::Output;

    fn midpoint(self, other: T) -> Self::Output {
        self.lerp(other, 0.5)
    }
}

pub trait Cross<T = Self> {
    type Output;

    fn cross(self, other: T) -> Self::Output;
}

impl<T> Interpolate for Duplet<T>
where
    T: Copy + Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: f64) -> Self::Output {
        Duplet(
            geometry::lerp(self.0, other.0, f),
            geometry::lerp(self.1, other.1, f),
        )
    }
}

impl<T> Interpolate for Triplet<T>
where
    T: Copy + Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: f64) -> Self::Output {
        Triplet(
            geometry::lerp(self.0, other.0, f),
            geometry::lerp(self.1, other.1, f),
            geometry::lerp(self.2, other.2, f),
        )
    }
}

#[cfg(feature = "geometry-cgmath")]
mod feature_geometry_cgmath {
    use cgmath::{ApproxEq, BaseFloat, BaseNum, EuclideanSpace, InnerSpace, Point2, Point3,
                 Vector2, Vector3};
    use num::NumCast;
    use std::ops::{AddAssign, MulAssign};

    use geometry;
    use geometry::ops::*;

    impl<T> Normalize for Vector3<T>
    where
        T: ApproxEq + BaseFloat + BaseNum,
    {
        #[inline(always)]
        fn normalize(self) -> Self {
            <Self as InnerSpace>::normalize(self)
        }
    }

    impl<T> Average for Point3<T>
    where
        T: AddAssign + BaseNum + MulAssign + NumCast,
    {
        fn average<I>(values: I) -> Self
        where
            I: IntoIterator<Item = Self>,
        {
            let values = values.into_iter().collect::<Vec<_>>();
            let n = <T as NumCast>::from(values.len()).unwrap();
            let sum = {
                let mut sum = Point3::origin();
                for point in values {
                    sum += Vector3::<T>::new(point.x, point.y, point.z);
                }
                sum
            };
            sum * (T::one() / n)
        }
    }

    impl<T> Interpolate for Point2<T>
    where
        T: BaseNum + NumCast,
    {
        type Output = Self;

        fn lerp(self, other: Self, f: f64) -> Self::Output {
            Point2::new(
                geometry::lerp(self.x, other.x, f),
                geometry::lerp(self.y, other.y, f),
            )
        }
    }

    impl<T> Interpolate for Point3<T>
    where
        T: BaseNum + NumCast,
    {
        type Output = Self;

        fn lerp(self, other: Self, f: f64) -> Self::Output {
            Point3::new(
                geometry::lerp(self.x, other.x, f),
                geometry::lerp(self.y, other.y, f),
                geometry::lerp(self.z, other.z, f),
            )
        }
    }

    impl<T> Interpolate for Vector2<T>
    where
        T: BaseNum + NumCast,
    {
        type Output = Self;

        fn lerp(self, other: Self, f: f64) -> Self::Output {
            Vector2::new(
                geometry::lerp(self.x, other.x, f),
                geometry::lerp(self.y, other.y, f),
            )
        }
    }

    impl<T> Interpolate for Vector3<T>
    where
        T: BaseNum + NumCast,
    {
        type Output = Self;

        fn lerp(self, other: Self, f: f64) -> Self::Output {
            Vector3::new(
                geometry::lerp(self.x, other.x, f),
                geometry::lerp(self.y, other.y, f),
                geometry::lerp(self.z, other.z, f),
            )
        }
    }

    impl<T> Cross for Vector3<T>
    where
        T: BaseFloat + BaseNum,
    {
        type Output = Self;

        #[inline(always)]
        fn cross(self, other: Self) -> Self::Output {
            Self::cross(self, other)
        }
    }
}

#[cfg(feature = "geometry-nalgebra")]
mod feature_geometry_nalgebra {
    use alga::general::Real;
    use nalgebra::{Point2, Point3, Scalar, Vector2, Vector3};
    use nalgebra::core::Matrix;
    use num::{Float, Num, NumCast};
    use std::ops::{AddAssign, MulAssign};

    use geometry;
    use geometry::ops::*;

    impl<T> Normalize for Vector3<T>
    where
        T: Float + Real + Scalar,
    {
        fn normalize(self) -> Self {
            Matrix::normalize(&self)
        }
    }

    // TODO: Implement `Average` for points and vectors of arbitrary dimension.
    impl<T> Average for Point3<T>
    where
        T: AddAssign + MulAssign + Num + NumCast + Scalar,
    {
        fn average<I>(values: I) -> Self
        where
            I: IntoIterator<Item = Self>,
        {
            let values = values.into_iter().collect::<Vec<_>>();
            let n = <T as NumCast>::from(values.len()).unwrap();
            let sum = {
                let mut sum = Point3::origin();
                for point in values {
                    sum += Vector3::<T>::new(point.x, point.y, point.z);
                }
                sum
            };
            sum * (T::one() / n)
        }
    }

    impl<T> Interpolate for Point2<T>
    where
        T: Num + NumCast + Scalar,
    {
        type Output = Self;

        fn lerp(self, other: Self, f: f64) -> Self::Output {
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

        fn lerp(self, other: Self, f: f64) -> Self::Output {
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

        fn lerp(self, other: Self, f: f64) -> Self::Output {
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

        fn lerp(self, other: Self, f: f64) -> Self::Output {
            Vector3::new(
                geometry::lerp(self.x, other.x, f),
                geometry::lerp(self.y, other.y, f),
                geometry::lerp(self.z, other.z, f),
            )
        }
    }

    impl<T> Cross for Vector3<T>
    where
        T: Float + Real + Scalar,
    {
        type Output = Self;

        fn cross(self, other: Self) -> Self::Output {
            Matrix::cross(&self, &other)
        }
    }
}
