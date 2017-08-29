use num::{self, Float, Num, NumCast};
use ordered_float::OrderedFloat;

use graph::{Attribute, Geometry};

pub trait Unit: Copy + Num {
    fn unit_radius() -> (Self, Self);
    fn unit_width() -> (Self, Self);
}

macro_rules! unit {
    (integer => $($t:ty),*) => {$(
        impl Unit for $t {
            fn unit_radius() -> (Self, Self) {
                use num::{One, Zero};
                (Self::zero(), Self::one() + Self::one())
            }

            fn unit_width() -> (Self, Self) {
                use num::{One, Zero};
                (Self::zero(), Self::one())
            }
        }
    )*};
    (real => $($t:ty),*) => {$(
        impl Unit for $t {
            fn unit_radius() -> (Self, Self) {
                use num::One;
                (-Self::one(), Self::one())
            }

            fn unit_width() -> (Self, Self) {
                use num::One;
                let half = Self::one() / (Self::one() + Self::one());
                (-half, half)
            }
        }
    )*};
}
unit!(integer => i8, i16, i32, i64, u8, u16, u32, u64);
unit!(real => f32, f64);

impl<T> Attribute for (T, T, T)
where
    T: Default + Unit,
{
}

impl<T> Attribute for (OrderedFloat<T>, OrderedFloat<T>, OrderedFloat<T>)
where
    T: Default + Float + Unit,
{
}

impl<T> Geometry for (T, T, T)
where
    T: Default + Unit,
{
    type Vertex = Self;
    type Edge = ();
    type Face = ();
}

impl<T> Geometry for (OrderedFloat<T>, OrderedFloat<T>, OrderedFloat<T>)
where
    T: Default + Float + Unit,
{
    type Vertex = Self;
    type Edge = ();
    type Face = ();
}

pub trait Interpolate<T = Self>: Sized {
    type Output;

    fn lerp(&self, other: &T, f: f64) -> Self::Output;

    fn midpoint(&self, other: &T) -> Self::Output {
        self.lerp(other, 0.5)
    }
}

impl<T> Interpolate for (T, T)
where
    T: Copy + Num + NumCast,
{
    type Output = Self;

    fn lerp(&self, other: &Self, f: f64) -> Self::Output {
        (lerp(self.0, other.0, f), lerp(self.1, other.1, f))
    }
}

impl<T> Interpolate for (T, T, T)
where
    T: Copy + Num + NumCast,
{
    type Output = Self;

    fn lerp(&self, other: &Self, f: f64) -> Self::Output {
        (
            lerp(self.0, other.0, f),
            lerp(self.1, other.1, f),
            lerp(self.2, other.2, f),
        )
    }
}

fn lerp<T>(a: T, b: T, f: f64) -> T
where
    T: Num + NumCast,
{
    let f = num::clamp(f, 0.0, 1.0);
    let af = <f64 as NumCast>::from(a).unwrap() * (1.0 - f);
    let bf = <f64 as NumCast>::from(b).unwrap() * f;
    <T as NumCast>::from(af + bf).unwrap()
}
