//! Incremental polygonal mesh construction.
//!
//! This module provides traits for incrementally constructing mesh data
//! structures. This API allows for meshes to be constructed in a way that is
//! agnostic to the specific data structure used to represent the mesh.
//!
//! [`Buildable`] is the primary trait of this API. It is implemented by mesh
//! data structures and exposes various associated types for their associated
//! data.  [`Buildable`] exposes a builder type via its
//! [`builder`][`Buildable::builder`] function. This builder type in turn
//! provides additional builders that can be used to construct a mesh from
//! _surfaces_ and _facets_.
//!
//! # Examples
//!
//! A function that generates a triangle from point geometry using builders:
//!
//! ```rust
//! # extern crate nalgebra;
//! # extern crate plexus;
//! #
//! use nalgebra::Point2;
//! use plexus::buffer::MeshBuffer3;
//! use plexus::builder::Buildable;
//! use plexus::geometry::FromGeometry;
//! use plexus::graph::MeshGraph;
//! use plexus::prelude::*;
//!
//! type E2 = Point2<f64>;
//!
//! fn trigon<B, T>(points: [T; 3]) -> Result<B, B::Error>
//! where
//!     B: Buildable,
//!     B::Vertex: FromGeometry<T>,
//! {
//!     let mut builder = B::builder();
//!     builder.surface_with(|builder| {
//!         let [a, b, c] = points;
//!         let a = builder.insert_vertex(a)?;
//!         let b = builder.insert_vertex(b)?;
//!         let c = builder.insert_vertex(c)?;
//!         builder.facets_with(|builder| builder.insert_facet(&[a, b, c], B::Facet::default()))
//!     })?;
//!     builder.build()
//! }
//!
//! // `MeshBuffer` and `MeshGraph` implement the `Buildable` trait.
//! let graph: MeshGraph<E2> = trigon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]).unwrap();
//! let buffer: MeshBuffer3<usize, E2> = trigon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]).unwrap();
//! ```
//!
//! [`Buildable::builder`]: crate::builder::Buildable::builder
//! [`Buildable`]: crate::builder::Buildable

// TODO: Is it useful to use a separate `FacetBuilder` type?
// TODO: Keys are not opaque. Especially for `MeshBuffer`, it may be possible to
//       "forge" keys. This could be prevented by using a wrapper type that is
//       not exported, but would introduce a performance cost to map and collect
//       slices of keys.

use std::fmt::Debug;
use std::hash::Hash;

use crate::geometry::FromGeometry;
use crate::transact::ClosedInput;

/// Polygonal mesh data structure that can be built incrementally.
///
/// This trait is the primary entrypoint into the builder API. Types that
/// implement this trait expose a [`MeshBuilder`] that can be used to construct
/// an instance of the type from surfaces and facets.
///
/// [`MeshBuilder`]: crate::builder::MeshBuilder
pub trait Buildable: Sized {
    type Builder: MeshBuilder<
        Commit = Self,
        Error = Self::Error,
        Vertex = Self::Vertex,
        Facet = Self::Facet,
    >;
    type Error: Debug;

    /// Vertex data.
    ///
    /// This type represents the data associated with vertices in the mesh.
    /// This typically includes positional data, but no data is required and
    /// this type may be the unit type `()`.
    ///
    /// Each builder trait also exposes such an associated type which is
    /// constrained by the `Builder` type.
    type Vertex;
    /// Facet data.
    ///
    /// This type represents the data associated with facets in the mesh. No
    /// data is required and this type may be the unit type `()`.
    ///
    /// Each builder trait also exposes such an associated type which is
    /// constrained by the `Builder` type.
    type Facet: Default;

    fn builder() -> Self::Builder;
}

/// Incremental polygonal mesh builder.
///
/// This trait exposes types that allow for mesh data structures to be
/// constructed incrementally from _surfaces_ and _facets_. A _surface_ is a
/// collection of vertices and facets connecting those vertices and typically
/// describes a _manifold_. A _facet_ is the connectivity between vertices in a
/// surface. Facets may also include associated data.
///
/// Construction is hierarchical, beginning with a surface and its vertices and
/// then facets. The association between a surface, its vertices, and facets is
/// enforced by the API, which accepts functions that operate on increasingly
/// specific builder types. The [`build`][`MeshBuilder::build`] function is used
/// to complete the construction of a mesh.
///
/// Builders may emit errors at any stage and errors depend on the
/// implementation of the builder types (and by extension the details of the
/// underlying data structure).
///
/// [`MeshBuilder::build`]: crate::builder::MeshBuilder::build
pub trait MeshBuilder: ClosedInput {
    type Builder: SurfaceBuilder<Error = Self::Error, Vertex = Self::Vertex, Facet = Self::Facet>;

    type Vertex;
    type Facet: Default;

    /// Constructs a surface.
    ///
    /// The given function is invoked with a [`SurfaceBuilder`], which can be
    /// used to insert vertices and construct facets.
    ///
    /// [`SurfaceBuilder`]: crate::builder::SurfaceBuilder
    fn surface_with<F, T, E>(&mut self, f: F) -> Result<T, Self::Error>
    where
        Self::Error: From<E>,
        F: FnOnce(&mut Self::Builder) -> Result<T, E>;

    /// Builds the mesh.
    ///
    /// The builder is consumed and a mesh with the constructed surfaces and
    /// facets is produced.
    ///
    /// # Errors
    ///
    /// Returns a latent error if the constructed surfaces and facets are
    /// incompatible with the underlying data structure. May return other
    /// errors depending on the details of the implementation.
    fn build(self) -> Result<Self::Commit, Self::Error> {
        self.commit().map_err(|(_, error)| error)
    }
}

pub trait SurfaceBuilder: ClosedInput {
    type Builder: FacetBuilder<Self::Key, Error = Self::Error, Facet = Self::Facet>;
    /// Vertex key.
    ///
    /// Each vertex is associated with a key of this type. This key is used to
    /// reference a given vertex and is required to insert faces with a
    /// [`FacetBuilder`].
    ///
    /// [`FacetBuilder`]: crate::builder::FacetBuilder
    type Key: Copy + Eq + Hash;

    type Vertex;
    type Facet: Default;

    /// Constructs facets in the surface.
    ///
    /// The given function is invoked with a [`FacetBuilder`], which can be used
    /// to insert facets.
    ///
    /// [`FacetBuilder`]: crate::builder::FacetBuilder
    fn facets_with<F, T, E>(&mut self, f: F) -> Result<T, Self::Error>
    where
        Self::Error: From<E>,
        F: FnOnce(&mut Self::Builder) -> Result<T, E>;

    /// Inserts a vertex into the surface.
    ///
    /// Returns a key that refers to the inserted vertex. This key can be used
    /// to insert facets with a [`FacetBuilder`].
    ///
    /// [`FacetBuilder`]: crate::builder::FacetBuilder
    fn insert_vertex<T>(&mut self, geometry: T) -> Result<Self::Key, Self::Error>
    where
        Self::Vertex: FromGeometry<T>;
}

pub trait FacetBuilder<K>: ClosedInput
where
    K: Eq + Hash,
{
    /// Facet key.
    ///
    /// Each facet is associated with a key of this type.
    type Key: Copy + Eq + Hash;

    type Facet: Default;

    /// Inserts a facet into the associated surface.
    ///
    /// A facet is formed from connectivity between vertices represented by an
    /// ordered slice of vertex keys from the associated [`SurfaceBuilder`].
    ///
    /// Returns a key that refers to the inserted facet.
    ///
    /// [`SurfaceBuilder`]: crate::builder::SurfaceBuilder
    fn insert_facet<T, U>(&mut self, keys: T, geometry: U) -> Result<Self::Key, Self::Error>
    where
        Self::Facet: FromGeometry<U>,
        T: AsRef<[K]>;
}
