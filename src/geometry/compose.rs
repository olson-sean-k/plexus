//! Higher order geometric traits.
//!
//! This module defines higher order traits for operations on a graph. It also
//! provides aliases for geometric types to improve readability of type
//! constraints. These traits can be used as contraints to prove to the
//! compiler that certain operations are supported without specifying
//! complicated relationships.
//!
//! The traits in this module have blanket implementations that apply when
//! certain geometric and operational traits are implemented. For example, if a
//! type implements `AsPosition` and the `Output` type of that implementation
//! also implements `Cross` and `Normalize`, then a `Geometry` using that type
//! as its `Vertex` attribute will likely implement the `FaceNormal` trait in
//! this module.
//!
//! # Examples
//!
//! A function that subdivides faces in a graph by splitting half-edges:
//!
//! ```rust
//! use plexus::geometry::alias::VertexPosition;
//! use plexus::geometry::compose::EdgeMidpoint;
//! use plexus::geometry::convert::AsPosition;
//! use plexus::geometry::Geometry;
//! use plexus::graph::{FaceView, GraphError, MeshGraph};
//! use plexus::prelude::*;
//!
//! // Requires `EdgeMidpoint` to split half-edges.
//! pub fn subdivide<G>(
//!     face: FaceView<&mut MeshGraph<G>, G>,
//! ) -> Result<FaceView<&mut MeshGraph<G>, G>, GraphError>
//! where
//!     G: EdgeMidpoint<Midpoint = VertexPosition<G>> + Geometry,
//!     G::Vertex: AsPosition,
//! {
//!     let arity = face.arity();
//!     let mut half = face.into_half();
//!     let mut splits = Vec::with_capacity(arity);
//!     for _ in 0..arity {
//!         let vertex = half.split()?;
//!         splits.push(vertex.key());
//!         half = vertex.into_outgoing_half().into_next_half();
//!     }
//!     let mut face = half.into_face().unwrap();
//!     for (a, b) in splits.into_iter().perimeter() {
//!         face = face.bisect(a, b)?.into_face().unwrap();
//!     }
//!     Ok(face)
//! }
//! ```

pub use crate::graph::geometry::*;
