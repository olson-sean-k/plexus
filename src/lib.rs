//! **Plexus** is a library for generating and manipulating 3D mesh data.
#![allow(unknown_lints)] // Allow clippy lints.

#[cfg(feature = "geometry-nalgebra")]
extern crate alga;
extern crate arrayvec;
extern crate itertools;
#[cfg(feature = "geometry-nalgebra")]
extern crate nalgebra;
extern crate num;
extern crate ordered_float;

use ordered_float::NotNaN;

// TODO: https://github.com/reem/rust-ordered-float/pull/28
pub type NotNan<T> = NotNaN<T>;

#[allow(non_camel_case_types)]
pub type r32 = NotNan<f32>;
#[allow(non_camel_case_types)]
pub type r64 = NotNan<f64>;

pub mod buffer;
pub mod generate;
pub mod graph;
