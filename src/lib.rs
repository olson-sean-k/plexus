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

use ordered_float::OrderedFloat;

#[allow(non_camel_case_types)]
pub type r32 = OrderedFloat<f32>;
#[allow(non_camel_case_types)]
pub type r64 = OrderedFloat<f64>;

pub mod buffer;
pub mod generate;
pub mod graph;
