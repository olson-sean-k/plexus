//! Morphisms between constant generics and numeric types.
//!
//! This module provides conversions between `typenum`'s unsigned integer types
//! and `usize` constant generics. These conversions are necessary to perform
//! static computations and comparisons, which cannot yet be done using constant
//! generics alone (e.g., `{N >= 3}`).
//!
//! See https://internals.rust-lang.org/t/const-generics-where-restrictions/12742/7

// TODO: Move this into the `theon` crate as part of its public API.

pub type ConstantOf<N> = <N as ToConstant>::Output;
pub type TypeOf<const N: usize> = <Constant<N> as ToType>::Output;

pub struct Constant<const N: usize>;

pub trait ToConstant {
    type Output;
}

pub trait ToType {
    type Output;
}

macro_rules! impl_morphisms {
    (types => $($n:ident),*$(,)?) => (
        use typenum::Unsigned;

        $(
            impl ToConstant for typenum::$n {
                type Output = Constant<{ typenum::$n::USIZE }>;
            }

            impl ToType for Constant<{ typenum::$n::USIZE }> {
                type Output = typenum::$n;
            }
        )*
    );
}
impl_morphisms!(types =>
    U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U21,
    U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32,
);
