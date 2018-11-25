use num::{Num, NumCast};
use std::ops::{Div, Mul};

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
    use cgmath::{
        ApproxEq, BaseFloat, BaseNum, EuclideanSpace, InnerSpace, Point2, Point3, Vector2, Vector3,
    };
    use num::NumCast;
    use std::ops::{AddAssign, MulAssign};

    use geometry;
    use geometry::ops::*;

    impl<T> Normalize for Vector2<T>
    where
        T: ApproxEq + BaseFloat + BaseNum,
    {
        fn normalize(self) -> Self {
            <Self as InnerSpace>::normalize(self)
        }
    }

    impl<T> Normalize for Vector3<T>
    where
        T: ApproxEq + BaseFloat + BaseNum,
    {
        fn normalize(self) -> Self {
            <Self as InnerSpace>::normalize(self)
        }
    }

    impl<T> Average for Point2<T>
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
                let mut sum = Point2::origin();
                for point in values {
                    sum += Vector2::<T>::new(point.x, point.y);
                }
                sum
            };
            sum * (T::one() / n)
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

    impl<T> Dot for Vector2<T>
    where
        T: BaseFloat + BaseNum,
    {
        type Output = T;

        fn dot(self, other: Self) -> Self::Output {
            <Self as InnerSpace>::dot(self, other)
        }
    }

    impl<T> Dot for Vector3<T>
    where
        T: BaseFloat + BaseNum,
    {
        type Output = T;

        fn dot(self, other: Self) -> Self::Output {
            <Self as InnerSpace>::dot(self, other)
        }
    }

    impl<T> Cross for Vector3<T>
    where
        T: BaseFloat + BaseNum,
    {
        type Output = Self;

        fn cross(self, other: Self) -> Self::Output {
            Self::cross(self, other)
        }
    }
}

#[cfg(feature = "geometry-mint")]
mod feature_geometry_mint {
    use mint::{Point2, Point3, Vector2, Vector3};
    use num::{Float, Num, NumCast};
    use std::ops::{AddAssign, MulAssign};

    use geometry;
    use geometry::ops::*;

    impl<T> Normalize for Vector2<T>
    where
        T: Float,
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
        T: Float,
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
        T: Clone + Num + NumCast,
    {
        fn average<I>(values: I) -> Self
        where
            I: IntoIterator<Item = Self>,
        {
            let values = values.into_iter().collect::<Vec<_>>();
            let n = <T as NumCast>::from(values.len()).unwrap();
            let sum = {
                let mut sum = Point2 {
                    x: T::zero(),
                    y: T::zero(),
                };
                for point in values {
                    sum = Point2 {
                        x: sum.x + point.x,
                        y: sum.y + point.y,
                    };
                }
                sum
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
        T: Clone + Num + NumCast,
    {
        fn average<I>(values: I) -> Self
        where
            I: IntoIterator<Item = Self>,
        {
            let values = values.into_iter().collect::<Vec<_>>();
            let n = <T as NumCast>::from(values.len()).unwrap();
            let sum = {
                let mut sum = Point3 {
                    x: T::zero(),
                    y: T::zero(),
                    z: T::zero(),
                };
                for point in values {
                    sum = Point3 {
                        x: sum.x + point.x,
                        y: sum.y + point.y,
                        z: sum.z + point.z,
                    };
                }
                sum
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

        fn lerp(self, other: Self, f: f64) -> Self::Output {
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

        fn lerp(self, other: Self, f: f64) -> Self::Output {
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

        fn lerp(self, other: Self, f: f64) -> Self::Output {
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

        fn lerp(self, other: Self, f: f64) -> Self::Output {
            Vector3 {
                x: geometry::lerp(self.x, other.x, f),
                y: geometry::lerp(self.y, other.y, f),
                z: geometry::lerp(self.z, other.z, f),
            }
        }
    }

    impl<T> Dot for Vector2<T>
    where
        T: AddAssign + Float + MulAssign,
    {
        type Output = T;

        fn dot(self, other: Self) -> Self::Output {
            (self.x * other.x) + (self.y * other.y)
        }
    }

    impl<T> Dot for Vector3<T>
    where
        T: AddAssign + Float + MulAssign,
    {
        type Output = T;

        fn dot(self, other: Self) -> Self::Output {
            (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
        }
    }

    impl<T> Cross for Vector3<T>
    where
        T: Float,
    {
        type Output = Self;

        fn cross(self, other: Self) -> Self::Output {
            Vector3 {
                x: (self.y * other.z) - (self.z * other.y),
                y: -((self.x * other.z) - (self.z * other.x)),
                z: (self.x * other.y) - (self.y * other.x),
            }
        }
    }
}

#[cfg(feature = "geometry-nalgebra")]
mod feature_geometry_nalgebra {
    use nalgebra::core::Matrix;
    use nalgebra::{Point2, Point3, Real, Scalar, Vector2, Vector3};
    use num::{Float, Num, NumCast};
    use std::ops::{AddAssign, MulAssign};

    use geometry;
    use geometry::ops::*;

    impl<T> Normalize for Vector2<T>
    where
        T: Float + Real + Scalar,
    {
        fn normalize(self) -> Self {
            Matrix::normalize(&self)
        }
    }

    impl<T> Normalize for Vector3<T>
    where
        T: Float + Real + Scalar,
    {
        fn normalize(self) -> Self {
            Matrix::normalize(&self)
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
            let values = values.into_iter().collect::<Vec<_>>();
            let n = <T as NumCast>::from(values.len()).unwrap();
            let sum = {
                let mut sum = Point2::origin();
                for point in values {
                    sum += Vector2::<T>::new(point.x, point.y);
                }
                sum
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

    impl<T> Dot for Vector2<T>
    where
        T: AddAssign + Float + MulAssign + Scalar,
    {
        type Output = T;

        fn dot(self, other: Self) -> Self::Output {
            Matrix::dot(&self, &other)
        }
    }

    impl<T> Dot for Vector3<T>
    where
        T: AddAssign + Float + MulAssign + Scalar,
    {
        type Output = T;

        fn dot(self, other: Self) -> Self::Output {
            Matrix::dot(&self, &other)
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
