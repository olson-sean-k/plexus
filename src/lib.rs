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

pub mod buffer;
pub mod generate;
pub mod graph;
pub mod ordered;

pub mod prelude {
    pub use generate::prelude::*;
}
