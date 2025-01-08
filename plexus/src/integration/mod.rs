//! Integration of external crates and foreign types.
//!
//! This module provides implementations of traits in Plexus for foreign types.
//! Integrated crates are re-exported within a sub-module, which can be used to
//! avoid versioning conflicts.
//!
//! The documentation for items re-exported from integrated crates is hidden.
//!
//! # Examples
//!
//! Using the integrated version of [`nalgebra`] instead of a direct dependency:
//!
//! ```rust
//! use plexus::integration::nalgebra; // Import `nalgebra` from `plexus`.
//!
//! use ::nalgebra::Point2;
//! use plexus::buffer::MeshBuffer;
//! use plexus::index::Flat3;
//! use plexus::prelude::*;
//!
//! let buffer = MeshBuffer::<Flat3, Point2<f64>>::from_raw_buffers(
//!     0..3usize,
//!     vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
//! )
//! .unwrap();
//! ```
//!
//! [`nalgebra`]: https://crates.io/crates/nalgebra

// TODO: Do not implement geometric traits and conversions over tuples of scalars. Prefer array
//       types instead.

mod cgmath;
mod glam;
mod mint;
mod nalgebra;
mod ultraviolet;
