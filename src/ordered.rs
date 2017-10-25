//! Ordering and hashing for floating point data.
//!
//! This module provides tools to ease the creation of geometry data that can
//! be ordered and hashed. For Plexus, this is most useful for quickly
//! identifying unique geometry in an iterator expression or `Mesh`, such as
//! using a `HashIndexer`.
//!
//! The [decorum](https://crates.io/crates/decorum) crate is used for ordering
//! and hashing floating point data. See that crate for details of its hashing
//! strategy and how it can be used to create hashable types containing
//! floating point data.
//!
//! # Examples
//!
//! Converting generator types into a hashable type via `HashConjugate`:
//!
//! ```rust
//! use plexus::generate::HashIndexer;
//! use plexus::generate::cube::Cube;
//! use plexus::prelude::*;
//!
//! let (indeces, positions) = Cube::<f32>::with_unit_width()
//!     .polygons_with_position()
//!     .map_vertices(|position| position.into_hash()) // Convert to hashable type.
//!     .triangulate()
//!     .index_vertices(HashIndexer::default());
//! ```

use std::hash::Hash;

// TODO: This module is limited and `HashConjugate` isn't too useful. Instead,
//       types like `NotNan`, `Finite`, `R32`, etc. can be used directly in
//       generators.

/// Provides conversion to and from a conjugate type that can be hashed.
///
/// This trait is primarily used to convert geometry to and from hashable data
/// for indexing.
pub trait HashConjugate: Sized {
    /// Conjugate type that provides `Eq` and `Hash` implementations.
    type Hash: Eq + Hash;

    /// Converts into the conjugate type.
    fn into_hash(self) -> Self::Hash;

    /// Converts from the conjugate type.
    fn from_hash(hash: Self::Hash) -> Self;
}
