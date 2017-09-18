//! Ordering and hashing for floating point data.
//!
//! This module provides tools to ease the creation of geometry data that can
//! be ordered and hashed. For Plexus, this is most useful for quickly
//! identifying unique geometry in an iterator expression or `Mesh`, such as
//! using a `HashIndexer`.
//!
//! This code is best used with the
//! [derivative](https://crates.io/crates/derivative) crate, which can be used
//! to specify a particular hashing function for a given field.
//!
//! # Examples
//!
//! Creating a basic vertex type that can be used for rendering and implements
//! `Hash`:
//!
//! ```no_run,rust
//! use plexus::ordered;
//!
//! // TODO: The `derive` and `derivative` attributes are commented out to prevent
//! //       build errors when running doc tests.
//! //#[derive(Derivative)]
//! //#[derivative(Hash)]
//! pub struct Vertex {
//!     //#[derivative(Hash(hash_with="ordered::hash_float_array"))]
//!     pub position: [f32; 3],
//! }
//! ```

use num::Float;
use ordered_float;
use std::hash::{Hash, Hasher};

// TODO: https://github.com/reem/rust-ordered-float/pull/28
pub type NotNan<T> = ordered_float::NotNaN<T>;
pub use ordered_float::OrderedFloat;

#[allow(non_camel_case_types)]
pub type r32 = NotNan<f32>;
#[allow(non_camel_case_types)]
pub type r64 = NotNan<f64>;

pub mod prelude {
    pub use ordered::HashConjugate;
}

pub trait HashConjugate: Sized {
    type Hash: Eq + Hash;

    fn into_hash(self) -> Self::Hash;
    fn from_hash(hash: Self::Hash) -> Self;
}

#[inline(always)]
pub fn hash_float<F, H>(f: &F, state: &mut H)
where
    F: Float,
    H: Hasher,
{
    OrderedFloat::from(*f).hash(state)
}

pub fn hash_float_slice<F, H>(slice: &[F], state: &mut H)
where
    F: Float,
    H: Hasher,
{
    for f in slice {
        hash_float(f, state);
    }
}

pub trait HashFloatArray {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher;
}

pub fn hash_float_array<F, H>(array: &F, state: &mut H)
where
    F: HashFloatArray,
    H: Hasher,
{
    array.hash(state);
}

macro_rules! hash_float_array {
    (lengths => $($N:expr),*) => {$(
        impl<T> HashFloatArray for [T; $N]
        where
            T: Float,
        {
            fn hash<H>(&self, state: &mut H)
            where
                H: Hasher
            {
                hash_float_slice(self, state)
            }
        }
    )*};
}
hash_float_array!(lengths => 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
