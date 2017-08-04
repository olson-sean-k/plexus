//! **Plexus** is a library for generating, manipulating, and buffering 3D mesh
//! data.

#![allow(unknown_lints)] // Allow clippy lints.

extern crate arrayvec;
extern crate num;
#[cfg(test)]
extern crate ordered_float;

pub mod buffer;
pub mod generate;
pub mod graph;
