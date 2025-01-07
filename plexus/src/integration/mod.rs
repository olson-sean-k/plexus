//! Integration of external crates and foreign types.
//!
//! This module provides implementations of traits in Plexus for foreign types.

// TODO: Do not implement geometric traits and conversions over tuples of scalars. Prefer array
//       types instead.

mod cgmath;
mod glam;
mod mint;
mod nalgebra;
mod ultraviolet;
