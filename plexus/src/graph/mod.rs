//! Half-edge graph representation of polygonal meshes.
//!
//! This module provides a flexible representation of polygonal meshes as a
//! [half-edge graph][dcel]. Plexus refers to _Half-edges_ and _edges_ as _arcs_
//! and _edges_, respectively. Graphs can store arbitrary data associated with
//! any topological entity (vertices, arcs, edges, and faces).
//!
//! Graph APIs support geometric operations if vertex data implements the
//! [`AsPosition`] trait.
//!
//! See the [user guide][guide-graphs] for more details and examples.
//!
//! # Representation
//!
//! A [`MeshGraph`] is fundamentally composed of four entities: _vertices_,
//! _arcs_, _edges_, and _faces_. The figure below summarizes the connectivity
//! of these entities.
//!
//! ![Half-Edge Graph Figure](https://plexus.rs/img/heg.svg)
//!
//! Arcs are directed and connect vertices. An arc that is directed toward a
//! vertex $A$ is an _incoming arc_ with respect to $A$. Similarly, an arc
//! directed away from such a vertex is an _outgoing arc_. Every vertex is
//! associated with exactly one _leading arc_, which is always an outgoing arc.
//! The vertex toward which an arc is directed is the arc's _destination vertex_
//! and the other is its _source vertex_.
//!
//! Every arc is paired with an _opposite arc_ with an opposing direction.
//! Given an arc from a vertex $A$ to a vertex $B$, that arc will have an
//! opposite arc from $B$ to $A$. Such arcs are notated $\overrightarrow{AB}$
//! and $\overrightarrow{BA}$. Together, these arcs form an _edge_, which is not
//! directed. An edge and its two arcs are together called a _composite edge_.
//!
//! Arcs are connected to their adjacent arcs, known as _next_ and _previous
//! arcs_. A traversal along a series of arcs is a _path_. The path formed by
//! traversing from an arc to its next arc and so on is a _ring_. When a face is
//! present within an ring, the arcs will refer to that face and the face will
//! refer to exactly one of the arcs in the ring (this is the leading arc of the
//! face). An arc with no associated face is known as a _boundary arc_.  If
//! either of an edge's arcs is a boundary arc, then that edge is a _boundary
//! edge_.
//!
//! A path that terminates is _open_ and a path that forms a loop is _closed_.
//! Rings are always closed. Paths may be notated using _sequence_ or _set
//! notation_ and both forms are used to describe rings and faces.
//!
//! Sequence notation is formed from the ordered sequence of vertices that a
//! path traverses, including the source vertex of the first arc and the
//! destination vertex of the last arc. Set notation is similar, but is
//! implicitly closed and only includes the ordered and unique set of vertices
//! traversed by the path. An open path over vertices $A$, $B$, and $C$ is
//! notated as a sequence $\overrightarrow{(A,B,C)}$. A closed path over
//! vertices $A$, $B$, and $C$ includes the arc $\overrightarrow{CA}$ and is
//! notated as a sequence $\overrightarrow{(A,B,C,A)}$ or a set
//! $\overrightarrow{\\{A,B,C\\}}$.
//!
//! Together with vertices and faces, the connectivity of arcs allows for
//! effecient traversals. For example, it becomes trivial to find adjacent
//! entities, such as the faces that share a given vertex or the adjacent faces
//! of a given face.
//!
//! [`MeshGraph`]s store entities using associative data structures with
//! strongly typed and opaque keys. These keys are used to refer entities in a
//! graph. Note that paths and rings are **not** entities and are not explicitly
//! stored in graphs.
//!
//! # Views
//!
//! [`MeshGraph`]s expose _views_ over their entities (vertices, arcs, edges,
//! and faces). Views are a type of _smart pointer_ and bind entity storage with
//! a key for a specific entity. They extend entities with rich behaviors and
//! expose their associated data via `get` and `get_mut` functions.
//!
//! Views provide the primary API for interacting with a [`MeshGraph`]'s
//! topology and data. There are three types of views summarized below:
//!
//! | Type      | Traversal | Exclusive | Data      | Topology  |
//! |-----------|-----------|-----------|-----------|-----------|
//! | Immutable | Yes       | No        | Immutable | Immutable |
//! | Mutable   | Yes       | Yes       | Mutable   | Mutable   |
//! | Orphan    | No        | No        | Mutable   | N/A       |
//!
//! _Immutable_ and _mutable views_ behave similarly to Rust's `&` and `&mut`
//! references: immutable views cannot mutate a graph and are not exclusive
//! while mutable views may mutate both the data and topology of a graph but are
//! exclusive.
//!
//! _Orphan views_ (simply referred to as _orphans_ in APIs) may mutate the data
//! of a graph, but they cannot access the topology of a graph and cannot
//! traverse a graph in any way. This is only useful for modifying the data in a
//! graph, but unlike mutable views, orphan views are not exclusive.
//!
//! Views perform _interior reborrows_, which reborrow the reference to storage
//! to construct other views. Immutable reborrows can be performed explicitly
//! using the conversions described below:
//!
//! | Function   | Receiver    | Borrow | Output    |
//! |------------|-------------|--------|-----------|
//! | `to_ref`   | `&self`     | `&_`   | Immutable |
//! | `into_ref` | `self`      | `&*_`  | Immutable |
//!
//! It is not possible to explicitly perform a mutable interior reborrow. Such a
//! reborrow could invalidate the source view by performing a topological
//! mutation. Mutable reborrows are performed beneath safe APIs, such as those
//! exposing iterators over orphan views.
//!
//! # Geometric Traits
//!
//! The [`GraphData`] trait is used to specify the types of data stored in
//! entities in a [`MeshGraph`]. If the `Vertex` data implements the
//! [`AsPosition`] trait and the positional data implements the appropriate
//! geometric traits, then geometric APIs like
//! [`split_at_midpoint`][`ArcView::split_at_midpoint`] and
//! [`poke_with_offset`][`FaceView::poke_with_offset`] can be used. Abstracting
//! this in generic code involves various traits from [`theon`].
//!
//! This module provides geometric traits that describe supported geometric
//! operations without the need to express complicated relationships between
//! types representing a [Euclidean space][`EuclideanSpace`]. These traits
//! express the geometric capabilites of [`GraphData`]. For example, the
//! following generic function requires [`EdgeMidpoint`] and subdivides faces in
//! a graph by splitting edges at their midpoints:
//!
//! ```rust
//! # extern crate plexus;
//! # extern crate smallvec;
//! #
//! use plexus::geometry::AsPositionMut;
//! use plexus::graph::{EdgeMidpoint, FaceView, GraphData, MeshGraph};
//! use plexus::prelude::*;
//! use smallvec::SmallVec;
//!
//! // Requires `EdgeMidpoint` for `split_at_midpoint`.
//! pub fn circumscribe<G>(face: FaceView<&mut MeshGraph<G>>) -> FaceView<&mut MeshGraph<G>>
//! where
//!     G: EdgeMidpoint + GraphData,
//!     G::Vertex: AsPositionMut,
//! {
//!     let arity = face.arity();
//!     let mut arc = face.into_arc();
//!     let mut splits = SmallVec::<[_; 4]>::with_capacity(arity);
//!     for _ in 0..arity {
//!         let vertex = arc.split_at_midpoint();
//!         splits.push(vertex.key());
//!         arc = vertex.into_outgoing_arc().into_next_arc();
//!     }
//!     let mut face = arc.into_face().unwrap();
//!     for (a, b) in splits.into_iter().perimeter() {
//!         face = face.split(ByKey(a), ByKey(b)).unwrap().into_face().unwrap();
//!     }
//!     face
//! }
//! ```
//!
//! # Examples
//!
//! Generating a [`MeshGraph`] from a [$uv$-sphere][`UvSphere`]:
//!
//! ```rust
//! # extern crate decorum;
//! # extern crate nalgebra;
//! # extern crate plexus;
//! #
//! use decorum::R64;
//! use nalgebra::Point3;
//! use plexus::graph::MeshGraph;
//! use plexus::prelude::*;
//! use plexus::primitive::generate::Position;
//! use plexus::primitive::sphere::UvSphere;
//!
//! type E3 = Point3<R64>;
//!
//! let mut graph: MeshGraph<E3> = UvSphere::default().polygons::<Position<E3>>().collect();
//! ```
//!
//! Extruding a face in a [`MeshGraph`]:
//!
//! ```rust
//! # extern crate decorum;
//! # extern crate nalgebra;
//! # extern crate plexus;
//! #
//! use decorum::R64;
//! use nalgebra::Point3;
//! use plexus::graph::MeshGraph;
//! use plexus::prelude::*;
//! use plexus::primitive::generate::Position;
//! use plexus::primitive::sphere::UvSphere;
//!
//! type E3 = Point3<R64>;
//!
//! let mut graph: MeshGraph<E3> = UvSphere::new(8, 8).polygons::<Position<E3>>().collect();
//! // Get the key of the first face and then extrude it.
//! let key = graph.faces().nth(0).unwrap().key();
//! let face = graph
//!     .face_mut(key)
//!     .unwrap()
//!     .extrude_with_offset(1.0)
//!     .unwrap();
//! ```
//!
//! Traversing and circulating over a [`MeshGraph`]:
//!
//! ```rust
//! # extern crate nalgebra;
//! # extern crate plexus;
//! #
//! use nalgebra::Point2;
//! use plexus::graph::MeshGraph;
//! use plexus::prelude::*;
//! use plexus::primitive::Tetragon;
//!
//! let mut graph = MeshGraph::<Point2<f64>>::from_raw_buffers(
//!     vec![Tetragon::new(0u32, 1, 2, 3)],
//!     vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
//! )
//! .unwrap();
//! graph.triangulate();
//!
//! // Traverse an arc and use a circulator to get the faces of a nearby vertex.
//! let key = graph.arcs().nth(0).unwrap().key();
//! let mut vertex = graph
//!     .arc_mut(key)
//!     .unwrap()
//!     .into_opposite_arc()
//!     .into_next_arc()
//!     .into_destination_vertex();
//! for mut face in vertex.adjacent_face_orphans() {
//!     // `face.get_mut()` provides a mutable reference to face data.
//! }
//! ```
//!
//! [dcel]: https://en.wikipedia.org/wiki/doubly_connected_edge_list
//! [guide-graphs]: https://plexus.rs/user-guide/graphs
//!
//! [`theon`]: https://crates.io/crates/theon
//!
//! [`Deref`]: std::ops::Deref
//! [`EuclideanSpace`]: theon::space::EuclideanSpace
//! [`AsPosition`]: crate::geometry::AsPosition
//! [`ArcView::split_at_midpoint`]: crate::graph::ArcView::split_at_midpoint
//! [`EdgeMidpoint`]: crate::graph::EdgeMidpoint
//! [`FaceView::poke_with_offset`]: crate::graph::FaceView::poke_with_offset
//! [`GraphData`]: crate::graph::GraphData
//! [`MeshGraph`]: crate::graph::MeshGraph
//! [`UvSphere`]: crate::primitive::sphere::UvSphere

mod builder;
mod core;
mod data;
mod edge;
mod face;
mod geometry;
mod mutation;
mod path;
mod vertex;

use decorum::cmp::IntrinsicOrd;
use decorum::R64;
use itertools::Itertools;
use num::{Integer, NumCast, ToPrimitive, Unsigned};
use smallvec::SmallVec;
use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::FromIterator;
use std::vec;
use theon::adjunct::{FromItems, Map};
use theon::query::Aabb;
use theon::space::{EuclideanSpace, Scalar};
use theon::{AsPosition, AsPositionMut};
use thiserror::Error;
use typenum::{self, NonZero};

use crate::buffer::{BufferError, FromRawBuffers, FromRawBuffersWithArity, MeshBuffer};
use crate::builder::{Buildable, FacetBuilder, MeshBuilder, SurfaceBuilder};
use crate::encoding::{FaceDecoder, FromEncoding, VertexDecoder};
use crate::entity::storage::{AsStorage, AsStorageMut, AsStorageOf, Fuse, Key, StorageObject};
use crate::entity::view::{Bind, Orphan, View};
use crate::entity::{Entity, EntityError};
use crate::geometry::{FromGeometry, IntoGeometry};
use crate::graph::builder::GraphBuilder;
use crate::graph::core::{Core, OwnedCore};
use crate::graph::data::Parametric;
use crate::graph::edge::{Arc, Edge};
use crate::graph::face::Face;
use crate::graph::mutation::face::FaceInsertCache;
use crate::graph::mutation::{Consistent, Immediate};
use crate::graph::vertex::Vertex;
use crate::index::{Flat, FromIndexer, Grouping, HashIndexer, IndexBuffer, IndexVertices, Indexer};
use crate::primitive::decompose::IntoVertices;
use crate::primitive::{IntoPolygons, Polygonal, UnboundedPolygon};
use crate::transact::Transact;
use crate::{DynamicArity, MeshArity, StaticArity};

pub use crate::entity::view::{ClosedView, Rebind};
pub use crate::graph::data::{EntityData, GraphData};
pub use crate::graph::edge::{ArcKey, ArcOrphan, ArcView, EdgeKey, EdgeOrphan, EdgeView, ToArc};
pub use crate::graph::face::{FaceKey, FaceOrphan, FaceView, Ring, ToRing};
pub use crate::graph::geometry::{
    ArcNormal, EdgeMidpoint, FaceCentroid, FaceNormal, FacePlane, VertexCentroid, VertexNormal,
    VertexPosition,
};
pub use crate::graph::path::Path;
pub use crate::graph::vertex::{VertexKey, VertexOrphan, VertexView};

pub use Selector::ByIndex;
pub use Selector::ByKey;

type Mutation<M> = mutation::Mutation<Immediate<M>>;

/// Errors concerning [`MeshGraph`]s.
///
/// [`MeshGraph`]: crate::graph::MeshGraph
#[derive(Debug, Error, PartialEq)]
pub enum GraphError {
    #[error("required topology not found")]
    TopologyNotFound,
    #[error("conflicting topology found")]
    TopologyConflict,
    #[error("topology malformed")]
    TopologyMalformed,
    #[error("arity is non-polygonal")]
    ArityNonPolygonal,
    /// The arity of a [`MeshGraph`] or other data structure is not compatible
    /// with an operation.
    #[error("conflicting arity; expected {expected}, but got {actual}")]
    ArityConflict {
        /// The expected arity.
        expected: usize,
        /// The incompatible arity that was encountered.
        actual: usize,
    },
    /// The compound arity of a [`MeshGraph`] or other data structure is not
    /// uniform.
    ///
    /// This error occurs when an operation requires a uniform arity but a graph
    /// or other data structure is non-uniform. See [`MeshArity`].
    ///
    /// [`MeshArity`]: crate::MeshArity
    #[error("arity is non-uniform")]
    ArityNonUniform,
    /// Geometry is incompatible or cannot be computed.
    #[error("geometric operation failed")]
    Geometry,
    /// A graph or other data structure is not compatible with an encoding.
    #[error("encoding operation failed")]
    Encoding,
}

// TODO: How should buffer errors be handled? Is this sufficient?
impl From<BufferError> for GraphError {
    fn from(error: BufferError) -> Self {
        match error {
            BufferError::ArityConflict { expected, actual } => {
                GraphError::ArityConflict { expected, actual }
            }
            _ => GraphError::Encoding,
        }
    }
}

impl From<EntityError> for GraphError {
    fn from(error: EntityError) -> Self {
        match error {
            EntityError::EntityNotFound => GraphError::TopologyNotFound,
            EntityError::Data => GraphError::Geometry,
        }
    }
}

trait OptionExt<T> {
    fn expect_consistent(self) -> T;
}

impl<T> OptionExt<T> for Option<T> {
    fn expect_consistent(self) -> T {
        self.expect("internal error: graph consistency violated")
    }
}

trait ResultExt<T, E> {
    fn expect_consistent(self) -> T
    where
        E: Debug;
}

impl<T, E> ResultExt<T, E> for Result<T, E> {
    fn expect_consistent(self) -> T
    where
        E: Debug,
    {
        self.expect("internal error: graph consistency violated")
    }
}

/// Entity selector.
///
/// Identifies an entity by key or index. Keys behave as an absolute selector
/// and uniquely identify a single entity within a [`MeshGraph`]. Indices behave
/// as a relative selector and identify an entity relative to some other entity.
/// `Selector` is used by operations that support both of these selection
/// mechanisms.
///
/// An index is typically used to select an adjacent entity or contained (and
/// ordered) entity, such as an adjacent face.
///
/// # Examples
///
/// Splitting a face by index (of its contained vertices):
///
/// ```rust
/// # extern crate decorum;
/// # extern crate nalgebra;
/// # extern crate plexus;
/// #
/// use decorum::R64;
/// use nalgebra::Point3;
/// use plexus::graph::MeshGraph;
/// use plexus::prelude::*;
/// use plexus::primitive::cube::Cube;
/// use plexus::primitive::generate::Position;
///
/// type E3 = Point3<R64>;
///
/// let mut graph: MeshGraph<E3> = Cube::new().polygons::<Position<E3>>().collect();
/// let key = graph.faces().nth(0).unwrap().key();
/// graph
///     .face_mut(key)
///     .unwrap()
///     .split(ByIndex(0), ByIndex(2))
///     .unwrap();
/// ```
///
/// [`MeshGraph`]: crate::graph::MeshGraph
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Selector<K> {
    ByKey(K),
    ByIndex(usize),
}

impl<K> Selector<K> {
    /// Gets the selector's key or passes its index to a function to resolve
    /// the key.
    pub fn key_or_else<E, F>(self, f: F) -> Result<K, GraphError>
    where
        E: Into<GraphError>,
        F: Fn(usize) -> Result<K, E>,
    {
        match self {
            Selector::ByKey(key) => Ok(key),
            Selector::ByIndex(index) => f(index).map_err(|error| error.into()),
        }
    }

    /// Gets the selector's index or passes its key to a function to resolve
    /// the index.
    pub fn index_or_else<E, F>(self, f: F) -> Result<usize, GraphError>
    where
        E: Into<GraphError>,
        F: Fn(K) -> Result<usize, E>,
    {
        match self {
            Selector::ByKey(key) => f(key).map_err(|error| error.into()),
            Selector::ByIndex(index) => Ok(index),
        }
    }
}

impl<K> From<K> for Selector<K>
where
    K: Key,
{
    fn from(key: K) -> Self {
        Selector::ByKey(key)
    }
}

impl<K> From<usize> for Selector<K> {
    fn from(index: usize) -> Self {
        Selector::ByIndex(index)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GraphKey {
    Vertex(VertexKey),
    Arc(ArcKey),
    Edge(EdgeKey),
    Face(FaceKey),
}

impl From<VertexKey> for GraphKey {
    fn from(key: VertexKey) -> Self {
        GraphKey::Vertex(key)
    }
}

impl From<ArcKey> for GraphKey {
    fn from(key: ArcKey) -> Self {
        GraphKey::Arc(key)
    }
}

impl From<EdgeKey> for GraphKey {
    fn from(key: EdgeKey) -> Self {
        GraphKey::Edge(key)
    }
}

impl From<FaceKey> for GraphKey {
    fn from(key: FaceKey) -> Self {
        GraphKey::Face(key)
    }
}

/// [Half-edge graph][dcel] representation of a polygonal mesh.
///
/// `MeshGraph`s form a polygonal mesh from four interconnected entities:
/// vertices, arcs, edges, and faces. These entities are exposed by view and
/// orphan types as well as types that represent rings and paths in a graph.
/// Entities can be associated with arbitrary data, including no data at all.
/// See the [`GraphData`] trait.
///
/// This flexible representation supports fast traversals and searches and can
/// be used to manipulate both the data and topology of a mesh.
///
/// See the [`graph`] module documentation and [user guide][guide-graphs] for
/// more details.
///
/// [dcel]: https://en.wikipedia.org/wiki/doubly_connected_edge_list
/// [guide-graphs]: https://plexus.rs/user-guide/graphs
///
/// [`GraphData`]: crate::graph::GraphData
/// [`graph`]: crate::graph
pub struct MeshGraph<G = (R64, R64, R64)>
where
    G: GraphData,
{
    core: OwnedCore<G>,
}

impl<G> MeshGraph<G>
where
    G: GraphData,
{
    /// Creates an empty `MeshGraph`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use plexus::graph::MeshGraph;
    ///
    /// let mut graph = MeshGraph::<()>::new();
    /// ```
    pub fn new() -> Self {
        MeshGraph::from(
            Core::empty()
                .fuse(<Vertex<G> as Entity>::Storage::default())
                .fuse(<Arc<G> as Entity>::Storage::default())
                .fuse(<Edge<G> as Entity>::Storage::default())
                .fuse(<Face<G> as Entity>::Storage::default()),
        )
    }

    /// Gets the number of vertices in the graph.
    pub fn vertex_count(&self) -> usize {
        self.as_storage_of::<Vertex<_>>().len()
    }

    /// Gets an immutable view of the vertex with the given key.
    pub fn vertex(&self, key: VertexKey) -> Option<VertexView<&Self>> {
        Bind::bind(self, key)
    }

    /// Gets a mutable view of the vertex with the given key.
    pub fn vertex_mut(&mut self, key: VertexKey) -> Option<VertexView<&mut Self>> {
        Bind::bind(self, key)
    }

    // TODO: Return `Clone + Iterator`.
    /// Gets an iterator of immutable views over the vertices in the graph.
    pub fn vertices(&self) -> impl Iterator<Item = VertexView<&Self>> {
        self.as_storage_of::<Vertex<_>>()
            .iter()
            .map(|(key, _)| key)
            .map(move |key| View::bind_unchecked(self, key))
            .map(From::from)
    }

    /// Gets an iterator of orphan views over the vertices in the graph.
    pub fn vertex_orphans(&mut self) -> impl Iterator<Item = VertexOrphan<G>> {
        self.as_storage_mut_of::<Vertex<_>>()
            .iter_mut()
            .map(|(key, data)| Orphan::bind_unchecked(data, key))
            .map(From::from)
    }

    /// Gets the number of arcs in the graph.
    pub fn arc_count(&self) -> usize {
        self.as_storage_of::<Arc<_>>().len()
    }

    /// Gets an immutable view of the arc with the given key.
    pub fn arc(&self, key: ArcKey) -> Option<ArcView<&Self>> {
        Bind::bind(self, key)
    }

    /// Gets a mutable view of the arc with the given key.
    pub fn arc_mut(&mut self, key: ArcKey) -> Option<ArcView<&mut Self>> {
        Bind::bind(self, key)
    }

    // TODO: Return `Clone + Iterator`.
    /// Gets an iterator of immutable views over the arcs in the graph.
    pub fn arcs(&self) -> impl Iterator<Item = ArcView<&Self>> {
        self.as_storage_of::<Arc<_>>()
            .iter()
            .map(|(key, _)| key)
            .map(move |key| View::bind_unchecked(self, key))
            .map(From::from)
    }

    /// Gets an iterator of orphan views over the arcs in the graph.
    pub fn arc_orphans(&mut self) -> impl Iterator<Item = ArcOrphan<G>> {
        self.as_storage_mut_of::<Arc<_>>()
            .iter_mut()
            .map(|(key, data)| Orphan::bind_unchecked(data, key))
            .map(From::from)
    }

    /// Gets the number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.as_storage_of::<Edge<_>>().len()
    }

    /// Gets an immutable view of the edge with the given key.
    pub fn edge(&self, key: EdgeKey) -> Option<EdgeView<&Self>> {
        Bind::bind(self, key)
    }

    /// Gets a mutable view of the edge with the given key.
    pub fn edge_mut(&mut self, key: EdgeKey) -> Option<EdgeView<&mut Self>> {
        Bind::bind(self, key)
    }

    // TODO: Return `Clone + Iterator`.
    /// Gets an iterator of immutable views over the edges in the graph.
    pub fn edges(&self) -> impl Iterator<Item = EdgeView<&Self>> {
        self.as_storage_of::<Edge<_>>()
            .iter()
            .map(|(key, _)| key)
            .map(move |key| View::bind_unchecked(self, key))
            .map(From::from)
    }

    /// Gets an iterator of orphan views over the edges in the graph.
    pub fn edge_orphans(&mut self) -> impl Iterator<Item = EdgeOrphan<G>> {
        self.as_storage_mut_of::<Edge<_>>()
            .iter_mut()
            .map(|(key, data)| Orphan::bind_unchecked(data, key))
            .map(From::from)
    }

    /// Gets the number of faces in the graph.
    pub fn face_count(&self) -> usize {
        self.as_storage_of::<Face<_>>().len()
    }

    /// Gets an immutable view of the face with the given key.
    pub fn face(&self, key: FaceKey) -> Option<FaceView<&Self>> {
        Bind::bind(self, key)
    }

    /// Gets a mutable view of the face with the given key.
    pub fn face_mut(&mut self, key: FaceKey) -> Option<FaceView<&mut Self>> {
        Bind::bind(self, key)
    }

    // TODO: Return `Clone + Iterator`.
    /// Gets an iterator of immutable views over the faces in the graph.
    pub fn faces(&self) -> impl Iterator<Item = FaceView<&Self>> {
        self.as_storage_of::<Face<_>>()
            .iter()
            .map(|(key, _)| key)
            .map(move |key| View::bind_unchecked(self, key))
            .map(From::from)
    }

    /// Gets an iterator of orphan views over the faces in the graph.
    pub fn face_orphans(&mut self) -> impl Iterator<Item = FaceOrphan<G>> {
        self.as_storage_mut_of::<Face<_>>()
            .iter_mut()
            .map(|(key, data)| Orphan::bind_unchecked(data, key))
            .map(From::from)
    }

    /// Gets an immutable path over the given sequence of vertex keys.
    ///
    /// # Errors
    ///
    /// Returns an error if a vertex is not found or the path is malformed.
    pub fn path<I>(&self, keys: I) -> Result<Path<&Self>, GraphError>
    where
        I: IntoIterator,
        I::Item: Borrow<VertexKey>,
    {
        Path::bind(self, keys)
    }

    /// Gets a mutable path over the given sequence of vertex keys.
    ///
    /// # Errors
    ///
    /// Returns an error if a vertex is not found or the path is malformed.
    pub fn path_mut<I>(&mut self, keys: I) -> Result<Path<&mut Self>, GraphError>
    where
        I: IntoIterator,
        I::Item: Borrow<VertexKey>,
    {
        Path::bind(self, keys)
    }

    /// Gets an axis-aligned bounding box that encloses the graph.
    pub fn aabb(&self) -> Aabb<VertexPosition<G>>
    where
        G::Vertex: AsPosition,
        VertexPosition<G>: EuclideanSpace,
        Scalar<VertexPosition<G>>: IntrinsicOrd,
    {
        Aabb::from_points(self.vertices().map(|vertex| *vertex.data.as_position()))
    }

    // TODO: This triangulation does not consider geometry and exhibits some
    //       bad behavior in certain situations. Triangulation needs to be
    //       reworked and may need to expose a bit more complexity. A geometric
    //       triangulation algorithm would be a useful addition and could
    //       detect concave faces and provide more optimal splits. See comments
    //       on `FaceView::triangulate`.
    /// Triangulates the graph, tessellating all faces into triangles.
    pub fn triangulate(&mut self) {
        // TODO: This implementation is a bit fragile and depends on the
        //       semantics of `TopologyConflict` in this context. It also panics
        //       if no valid split is found given all offsets or if some other
        //       error is encountered while splitting. Can this code assume that
        //       any of these conditions aren't possible? This should work a bit
        //       better than using `FaceView::triangulate` until triangulation
        //       is reworked.
        let keys = self
            .as_storage_of::<Face<_>>()
            .iter()
            .map(|(key, _)| key)
            .collect::<Vec<_>>();
        for key in keys {
            let mut face = self.face_mut(key).unwrap();
            let mut offset = 0;
            while face.arity() > 3 {
                match face.split(ByIndex(offset), ByIndex(offset + 2)) {
                    Ok(next) => {
                        face = next.into_face().expect_consistent();
                        offset = 0;
                    }
                    Err(GraphError::TopologyConflict) => {
                        // Retry if the split intersected another face. See
                        // `FaceSplitCache::from_face`.
                        face = self.face_mut(key).unwrap();
                        offset += 1;
                        if offset >= face.arity() {
                            panic!()
                        }
                    }
                    _ => panic!(),
                }
            }
        }
    }

    /// Smooths the positions of vertices in the graph.
    ///
    /// Each position is translated by its offset from its centroid scaled by
    /// the given factor. The centroid of a vertex position is the mean of the
    /// positions of its adjacent vertices. That is, given a factor $k$ and a
    /// vertex with position $P$ and centroid $Q$, its position becomes
    /// $P+k(Q-P)$.
    pub fn smooth<T>(&mut self, factor: T)
    where
        T: Into<Scalar<VertexPosition<G>>>,
        G: VertexCentroid,
        G::Vertex: AsPositionMut,
        VertexPosition<G>: EuclideanSpace,
    {
        let factor = factor.into();
        let mut positions = HashMap::with_capacity(self.vertex_count());
        for vertex in self.vertices() {
            let position = *vertex.position();
            positions.insert(
                vertex.key(),
                position + ((vertex.centroid() - position) * factor),
            );
        }
        for mut vertex in self.vertex_orphans() {
            *vertex.get_mut().as_position_mut() = positions.remove(&vertex.key()).unwrap();
        }
    }

    /// Splits the graph along a path.
    ///
    /// Splitting a graph creates boundaries along the given path and copies any
    /// necessary vertex, arc, and edge geometry.
    ///
    /// If the path bisects the graph, then splitting will result in disjointed
    /// sub-graphs.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use nalgebra::Point2;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::Trigon;
    ///
    /// type E2 = Point2<f64>;
    ///
    /// // Create a graph from two triangles.
    /// let mut graph = MeshGraph::<E2>::from_raw_buffers(
    ///     vec![Trigon::new(0usize, 1, 2), Trigon::new(2, 1, 3)],
    ///     vec![
    ///         (-1.0, 0.0),
    ///         (0.0, -1.0),
    ///         (0.0, 1.0),
    ///         (1.0, 0.0),
    ///     ],
    /// )
    /// .unwrap();
    ///
    /// // Find the shared edge that bisects the triangles and then construct a path
    /// // along the edge and split the graph.
    /// let key = graph
    ///     .edges()
    ///     .find(|edge| !edge.is_boundary_edge())
    ///     .map(|edge| edge.into_arc().key())
    ///     .unwrap();
    /// let mut path = graph.arc_mut(key).unwrap().into_path();
    /// MeshGraph::split_at_path(path).unwrap();
    /// ```
    pub fn split_at_path(path: Path<&mut Self>) -> Result<(), GraphError> {
        let _ = path;
        unimplemented!()
    }

    /// Gets an iterator over a vertex within each disjoint sub-graph.
    ///
    /// Traverses the graph and returns an arbitrary vertex within each
    /// _disjoint sub-graph_. A sub-graph is _disjoint_ if it cannot be reached
    /// from all other topology in the graph.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use nalgebra::Point2;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::Trigon;
    ///
    /// type E2 = Point2<f64>;
    ///
    /// // Create a graph from two disjoint triangles.
    /// let graph = MeshGraph::<E2>::from_raw_buffers(
    ///     vec![Trigon::new(0u32, 1, 2), Trigon::new(3, 4, 5)],
    ///     vec![
    ///         (-2.0, 0.0),
    ///         (-1.0, 0.0),
    ///         (-1.0, 1.0),
    ///         (1.0, 0.0),
    ///         (2.0, 0.0),
    ///         (1.0, 1.0),
    ///     ],
    /// )
    /// .unwrap();
    ///
    /// // A vertex from each disjoint triangle is returned.
    /// for vertex in graph.disjoint_subgraph_vertices() {
    ///     // ...
    /// }
    /// ```
    pub fn disjoint_subgraph_vertices(&self) -> impl ExactSizeIterator<Item = VertexView<&Self>> {
        let keys = self
            .as_storage_of::<Vertex<_>>()
            .iter()
            .map(|(key, _)| key)
            .collect::<HashSet<_>>();
        let mut subkeys = HashSet::with_capacity(self.vertex_count());
        let mut vertices = SmallVec::<[VertexView<_>; 4]>::new();
        while let Some(key) = keys.difference(&subkeys).nth(0) {
            let vertex = VertexView::from(View::bind_unchecked(self, *key));
            vertices.push(vertex);
            subkeys.extend(vertex.traverse_by_depth().map(|vertex| vertex.key()));
        }
        vertices.into_iter()
    }

    /// Moves disjoint sub-graphs into separate graphs.
    pub fn into_disjoint_subgraphs(self) -> Vec<Self> {
        unimplemented!()
    }

    /// Creates a [`Buildable`] mesh data structure from the graph.
    ///
    /// The output is created from each unique vertex in the graph. No face data
    /// is used, and the `Facet` type is always the unit type `()`.
    ///
    /// # Examples
    ///
    /// Creating a [`MeshBuffer`] from a [`MeshGraph`] used to modify a cube:
    ///
    /// ```rust
    /// # extern crate decorum;
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use decorum::N64;
    /// use nalgebra::Point3;
    /// use plexus::buffer::MeshBufferN;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::generate::Position;
    ///
    /// type E3 = Point3<N64>;
    ///
    /// let mut graph: MeshGraph<E3> = Cube::new().polygons::<Position<E3>>().collect();
    /// let key = graph.faces().nth(0).unwrap().key();
    /// graph
    ///     .face_mut(key)
    ///     .unwrap()
    ///     .extrude_with_offset(1.0)
    ///     .unwrap();
    ///
    /// let buffer: MeshBufferN<usize, E3> = graph.to_mesh_by_vertex().unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the graph does not have constant arity that is
    /// compatible with the index buffer. Typically, a graph is triangulated
    /// before being converted to a buffer.
    ///
    /// [`MeshBuffer`]: crate::buffer::MeshBuffer
    /// [`Buildable`]: crate::builder::Buildable
    /// [`MeshGraph`]: crate::graph::MeshGraph
    pub fn to_mesh_by_vertex<B>(&self) -> Result<B, B::Error>
    where
        B: Buildable<Facet = ()>,
        B::Vertex: FromGeometry<G::Vertex>,
    {
        self.to_mesh_by_vertex_with(|vertex| vertex.data.into_geometry())
    }

    /// Creates a [`Buildable`] mesh data structure from the graph.
    ///
    /// The output is created from each unique vertex in the graph, which is
    /// converted by the given function. No face data is used, and the `Facet`
    /// type is always the unit type `()`.
    ///
    /// # Errors
    ///
    /// Returns an error if the vertex geometry cannot be inserted into the
    /// output, there are arity conflicts, or the output does not support
    /// topology found in the graph.
    ///
    /// [`Buildable`]: crate::builder::Buildable
    pub fn to_mesh_by_vertex_with<B, F>(&self, mut f: F) -> Result<B, B::Error>
    where
        B: Buildable<Facet = ()>,
        F: FnMut(VertexView<&Self>) -> B::Vertex,
    {
        let mut builder = B::builder();
        builder.surface_with(|builder| {
            let mut keys = HashMap::with_capacity(self.vertex_count());
            for vertex in self.vertices() {
                keys.insert(vertex.key(), builder.insert_vertex(f(vertex))?);
            }
            builder.facets_with(|builder| {
                for face in self.faces() {
                    let indices = face
                        .adjacent_vertices()
                        .map(|vertex| keys[&vertex.key()])
                        .collect::<SmallVec<[_; 8]>>();
                    builder.insert_facet(indices.as_slice(), ())?;
                }
                Ok(())
            })
        })?;
        builder.build()
    }

    /// Creates a [`Buildable`] mesh data structure from the graph.
    ///
    /// The output is created from each face in the graph. For each face, the
    /// face data and data for each of its vertices is inserted into the mesh
    /// via [`FromGeometry`]. This means that a vertex is inserted for each of
    /// its adjacent faces.
    ///
    /// # Errors
    ///
    /// Returns an error if the vertex geometry cannot be inserted into the
    /// output, there are arity conflicts, or the output does not support
    /// topology found in the graph.
    ///
    /// [`Buildable`]: crate::builder::Buildable
    /// [`FromGeometry`]: crate::geometry::FromGeometry
    pub fn to_mesh_by_face<B>(&self) -> Result<B, B::Error>
    where
        B: Buildable,
        B::Vertex: FromGeometry<G::Vertex>,
        B::Facet: FromGeometry<G::Face>,
    {
        self.to_mesh_by_face_with(|_, vertex| vertex.data.into_geometry())
    }

    /// Creates a [`Buildable`] mesh data structure from the graph.
    ///
    /// The output is created from each face in the graph. For each face, the
    /// face data and data for each of its vertices is converted into the output
    /// vertex data by the given function. This means that a vertex is inserted
    /// for each of its adjacent faces. The data of each face is is inserted
    /// into the output via [`FromGeometry`].
    ///
    /// # Examples
    ///
    /// Creating a [`MeshBuffer`] from a [`MeshGraph`] used to compute normals:
    ///
    /// ```rust
    /// # extern crate decorum;
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use decorum::R64;
    /// use nalgebra::Point3;
    /// use plexus::buffer::MeshBuffer;
    /// use plexus::geometry::Vector;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::generate::Position;
    /// use plexus::primitive::BoundedPolygon;
    ///
    /// type E3 = Point3<R64>;
    ///
    /// pub struct Vertex {
    ///     pub position: E3,
    ///     pub normal: Vector<E3>,
    /// }
    ///
    /// let graph: MeshGraph<E3> = Cube::new().polygons::<Position<E3>>().collect();
    ///
    /// let buffer: MeshBuffer<BoundedPolygon<usize>, _> = graph
    ///     .to_mesh_by_face_with(|face, vertex| Vertex {
    ///         position: *vertex.position(),
    ///         normal: face.normal().unwrap(),
    ///     })
    ///     .unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the vertex geometry cannot be inserted into the
    /// output, there are arity conflicts, or the output does not support
    /// topology found in the graph.
    ///
    /// [`MeshBuffer`]: crate::buffer::MeshBuffer
    /// [`Buildable`]: crate::builder::Buildable
    /// [`FromGeometry`]: crate::geometry::FromGeometry
    /// [`MeshGraph`]: crate::graph::MeshGraph
    pub fn to_mesh_by_face_with<B, F>(&self, mut f: F) -> Result<B, B::Error>
    where
        B: Buildable,
        B::Facet: FromGeometry<G::Face>,
        F: FnMut(FaceView<&Self>, VertexView<&Self>) -> B::Vertex,
    {
        let mut builder = B::builder();
        builder.surface_with(|builder| {
            for face in self.faces() {
                let indices = face
                    .adjacent_vertices()
                    .map(|vertex| builder.insert_vertex(f(face, vertex)))
                    .collect::<Result<SmallVec<[_; 8]>, _>>()?;
                builder
                    .facets_with(|builder| builder.insert_facet(indices.as_slice(), face.data))?;
            }
            Ok(())
        })?;
        builder.build()
    }
}

impl<G> AsStorage<Vertex<G>> for MeshGraph<G>
where
    G: GraphData,
{
    fn as_storage(&self) -> &StorageObject<Vertex<G>> {
        self.core.as_storage_of::<Vertex<_>>()
    }
}

impl<G> AsStorage<Arc<G>> for MeshGraph<G>
where
    G: GraphData,
{
    fn as_storage(&self) -> &StorageObject<Arc<G>> {
        self.core.as_storage_of::<Arc<_>>()
    }
}

impl<G> AsStorage<Edge<G>> for MeshGraph<G>
where
    G: GraphData,
{
    fn as_storage(&self) -> &StorageObject<Edge<G>> {
        self.core.as_storage_of::<Edge<_>>()
    }
}

impl<G> AsStorage<Face<G>> for MeshGraph<G>
where
    G: GraphData,
{
    fn as_storage(&self) -> &StorageObject<Face<G>> {
        self.core.as_storage_of::<Face<_>>()
    }
}

impl<G> AsStorageMut<Vertex<G>> for MeshGraph<G>
where
    G: GraphData,
{
    fn as_storage_mut(&mut self) -> &mut StorageObject<Vertex<G>> {
        self.core.as_storage_mut_of::<Vertex<_>>()
    }
}

impl<G> AsStorageMut<Arc<G>> for MeshGraph<G>
where
    G: GraphData,
{
    fn as_storage_mut(&mut self) -> &mut StorageObject<Arc<G>> {
        self.core.as_storage_mut_of::<Arc<_>>()
    }
}

impl<G> AsStorageMut<Edge<G>> for MeshGraph<G>
where
    G: GraphData,
{
    fn as_storage_mut(&mut self) -> &mut StorageObject<Edge<G>> {
        self.core.as_storage_mut_of::<Edge<_>>()
    }
}

impl<G> AsStorageMut<Face<G>> for MeshGraph<G>
where
    G: GraphData,
{
    fn as_storage_mut(&mut self) -> &mut StorageObject<Face<G>> {
        self.core.as_storage_mut_of::<Face<_>>()
    }
}

/// Exposes a [`MeshBuilder`] that can be used to construct a [`MeshGraph`]
/// incrementally from _surfaces_ and _facets_.
///
/// See the [`builder`] module documentation for more.
///
/// # Examples
///
/// Creating a [`MeshGraph`] from a triangle:
///
/// ```rust
/// # extern crate nalgebra;
/// # extern crate plexus;
/// #
/// use nalgebra::Point2;
/// use plexus::builder::Buildable;
/// use plexus::graph::MeshGraph;
/// use plexus::prelude::*;
///
/// let mut builder = MeshGraph::<Point2<f64>>::builder();
/// let graph = builder
///     .surface_with(|builder| {
///         let a = builder.insert_vertex((0.0, 0.0))?;
///         let b = builder.insert_vertex((1.0, 0.0))?;
///         let c = builder.insert_vertex((0.0, 1.0))?;
///         builder.facets_with(|builder| builder.insert_facet(&[a, b, c], ()))
///     })
///     .and_then(|_| builder.build())
///     .unwrap();
/// ```
///
/// [`MeshBuilder`]: crate::builder::MeshBuilder
/// [`builder`]: crate::builder
/// [`MeshGraph`]: crate::graph::MeshGraph
impl<G> Buildable for MeshGraph<G>
where
    G: GraphData,
{
    type Builder = GraphBuilder<G>;
    type Error = GraphError;

    type Vertex = G::Vertex;
    type Facet = G::Face;

    fn builder() -> Self::Builder {
        Default::default()
    }
}

impl<G> Consistent for MeshGraph<G> where G: GraphData {}

impl<G> Default for MeshGraph<G>
where
    G: GraphData,
{
    fn default() -> Self {
        MeshGraph::new()
    }
}

impl<G> DynamicArity for MeshGraph<G>
where
    G: GraphData,
{
    type Dynamic = MeshArity;

    fn arity(&self) -> Self::Dynamic {
        MeshArity::from_components::<FaceView<_>, _>(self.faces())
    }
}

impl<P, G> From<P> for MeshGraph<G>
where
    P: Polygonal,
    G: GraphData,
    G::Vertex: FromGeometry<P::Vertex>,
{
    fn from(polygon: P) -> Self {
        let arity = polygon.arity();
        MeshGraph::from_raw_buffers_with_arity(0..arity, polygon, arity)
            .expect("inconsistent polygon")
    }
}

impl<G> From<OwnedCore<G>> for MeshGraph<G>
where
    G: GraphData,
{
    fn from(core: OwnedCore<G>) -> Self {
        MeshGraph { core }
    }
}

impl<E, G> FromEncoding<E> for MeshGraph<G>
where
    E: FaceDecoder + VertexDecoder,
    G: GraphData,
    G::Face: FromGeometry<E::Face>,
    G::Vertex: FromGeometry<E::Vertex>,
{
    type Error = GraphError;

    fn from_encoding(
        vertices: <E as VertexDecoder>::Output,
        faces: <E as FaceDecoder>::Output,
    ) -> Result<Self, Self::Error> {
        let mut mutation = Mutation::from(MeshGraph::new());
        let keys = vertices
            .into_iter()
            .map(|geometry| mutation::vertex::insert(&mut mutation, geometry.into_geometry()))
            .collect::<Vec<_>>();
        for (perimeter, geometry) in faces {
            let perimeter = perimeter
                .into_iter()
                .map(|index| keys[index])
                .collect::<SmallVec<[_; 4]>>();
            let cache = FaceInsertCache::from_storage(&mutation, perimeter.as_slice())?;
            let geometry = geometry.into_geometry();
            mutation::face::insert_with(&mut mutation, cache, || (Default::default(), geometry))?;
        }
        mutation.commit().map_err(|(_, error)| error)
    }
}

impl<G, P> FromIndexer<P, P> for MeshGraph<G>
where
    G: GraphData,
    G::Vertex: FromGeometry<P::Vertex>,
    P: Map<usize> + Polygonal,
    P::Output: Grouping<Group = P::Output> + IntoVertices + Polygonal<Vertex = usize>,
    Vec<P::Output>: IndexBuffer<P::Output, Index = usize>,
{
    type Error = GraphError;

    // TODO: This appears to be a false positive. The `collect` is necessary,
    //       because the data is transformed and read randomly by index. See
    //       https://github.com/rust-lang/rust-clippy/issues/5991
    #[allow(clippy::needless_collect)]
    fn from_indexer<I, N>(input: I, indexer: N) -> Result<Self, Self::Error>
    where
        I: IntoIterator<Item = P>,
        N: Indexer<P, P::Vertex>,
    {
        let mut mutation = Mutation::from(MeshGraph::new());
        let (indices, vertices) = input.into_iter().index_vertices(indexer);
        let vertices = vertices
            .into_iter()
            .map(|vertex| mutation::vertex::insert(&mut mutation, vertex.into_geometry()))
            .collect::<Vec<_>>();
        for face in indices {
            let perimeter = face
                .into_vertices()
                .into_iter()
                .map(|index| vertices[index])
                .collect::<SmallVec<[_; 4]>>();
            let cache = FaceInsertCache::from_storage(&mutation, &perimeter)?;
            mutation::face::insert_with(&mut mutation, cache, Default::default)?;
        }
        mutation.commit().map_err(|(_, error)| error)
    }
}

impl<G, P> FromIterator<P> for MeshGraph<G>
where
    G: GraphData,
    G::Vertex: FromGeometry<P::Vertex>,
    P: Polygonal,
    P::Vertex: Clone + Eq + Hash,
    Self: FromIndexer<P, P>,
{
    fn from_iter<I>(input: I) -> Self
    where
        I: IntoIterator<Item = P>,
    {
        Self::from_indexer(input, HashIndexer::default()).unwrap_or_else(|_| Self::default())
    }
}

impl<P, G, H> FromRawBuffers<P, H> for MeshGraph<G>
where
    P: IntoVertices + Polygonal,
    P::Vertex: Integer + ToPrimitive + Unsigned,
    G: GraphData,
    G::Vertex: FromGeometry<H>,
{
    type Error = GraphError;

    fn from_raw_buffers<I, J>(indices: I, vertices: J) -> Result<Self, Self::Error>
    where
        I: IntoIterator<Item = P>,
        J: IntoIterator<Item = H>,
    {
        let mut mutation = Mutation::from(MeshGraph::new());
        let vertices = vertices
            .into_iter()
            .map(|vertex| mutation::vertex::insert(&mut mutation, vertex.into_geometry()))
            .collect::<Vec<_>>();
        for face in indices {
            let mut perimeter = SmallVec::<[_; 4]>::with_capacity(face.arity());
            for index in face.into_vertices() {
                let index = <usize as NumCast>::from(index).unwrap();
                perimeter.push(
                    *vertices
                        .get(index)
                        .ok_or_else(|| GraphError::TopologyNotFound)?,
                );
            }
            let cache = FaceInsertCache::from_storage(&mutation, &perimeter)?;
            mutation::face::insert_with(&mut mutation, cache, Default::default)?;
        }
        mutation.commit().map_err(|(_, error)| error)
    }
}

impl<N, G, H> FromRawBuffersWithArity<N, H> for MeshGraph<G>
where
    N: Integer + ToPrimitive + Unsigned,
    G: GraphData,
    G::Vertex: FromGeometry<H>,
{
    type Error = GraphError;

    /// Creates a [`MeshGraph`] from [raw buffers][`buffer`]. The arity of the
    /// polygons in the index buffer must be given and constant.
    ///
    /// # Errors
    ///
    /// Returns an error if the arity of the index buffer is not constant, any
    /// index is out of bounds, or there is an error inserting topology into the
    /// graph.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use nalgebra::Point3;
    /// use plexus::graph::MeshGraph;
    /// use plexus::index::{Flat3, LruIndexer};
    /// use plexus::prelude::*;
    /// use plexus::primitive::generate::Position;
    /// use plexus::primitive::sphere::UvSphere;
    ///
    /// type E3 = Point3<f64>;
    ///
    /// let (indices, positions) = UvSphere::new(16, 16)
    ///     .polygons::<Position<E3>>()
    ///     .triangulate()
    ///     .index_vertices::<Flat3, _>(LruIndexer::with_capacity(256));
    /// let mut graph = MeshGraph::<E3>::from_raw_buffers_with_arity(indices, positions, 3).unwrap();
    /// ```
    ///
    /// [`buffer`]: crate::buffer
    /// [`MeshGraph`]: crate::graph::MeshGraph
    fn from_raw_buffers_with_arity<I, J>(
        indices: I,
        vertices: J,
        arity: usize,
    ) -> Result<Self, Self::Error>
    where
        I: IntoIterator<Item = N>,
        J: IntoIterator<Item = H>,
    {
        if arity < 3 {
            return Err(GraphError::ArityNonPolygonal);
        }
        let mut mutation = Mutation::from(MeshGraph::new());
        let vertices = vertices
            .into_iter()
            .map(|vertex| mutation::vertex::insert(&mut mutation, vertex.into_geometry()))
            .collect::<Vec<_>>();
        for face in &indices
            .into_iter()
            .map(|index| <usize as NumCast>::from(index).unwrap())
            .chunks(arity)
        {
            let face = face.collect::<Vec<_>>();
            if face.len() != arity {
                // Index buffer length is not a multiple of arity.
                return Err(GraphError::ArityConflict {
                    expected: arity,
                    actual: face.len(),
                });
            }
            let mut perimeter = SmallVec::<[_; 4]>::with_capacity(arity);
            for index in face {
                perimeter.push(
                    *vertices
                        .get(index)
                        .ok_or_else(|| GraphError::TopologyNotFound)?,
                );
            }
            let cache = FaceInsertCache::from_storage(&mutation, &perimeter)?;
            mutation::face::insert_with(&mut mutation, cache, Default::default)?;
        }
        mutation.commit().map_err(|(_, error)| error)
    }
}

impl<G> Parametric for MeshGraph<G>
where
    G: GraphData,
{
    type Data = G;
}

impl<G> Into<OwnedCore<G>> for MeshGraph<G>
where
    G: GraphData,
{
    fn into(self) -> OwnedCore<G> {
        let MeshGraph { core, .. } = self;
        core
    }
}

impl<G> IntoPolygons for MeshGraph<G>
where
    G: GraphData,
{
    type Output = vec::IntoIter<Self::Polygon>;
    type Polygon = UnboundedPolygon<G::Vertex>;

    fn into_polygons(self) -> Self::Output {
        self.faces()
            .map(|face| {
                // The arity of a face in a graph must be polygonal (three or
                // higher) so this should never fail.
                let vertices = face.adjacent_vertices().map(|vertex| vertex.data);
                UnboundedPolygon::from_items(vertices).expect_consistent()
            })
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl<G> StaticArity for MeshGraph<G>
where
    G: GraphData,
{
    type Static = (usize, Option<usize>);

    const ARITY: Self::Static = (3, None);
}

impl<A, N, H, G> TryFrom<MeshBuffer<Flat<A, N>, H>> for MeshGraph<G>
where
    A: NonZero + typenum::Unsigned,
    N: Copy + Integer + NumCast + Unsigned,
    H: Clone,
    G: GraphData,
    G::Vertex: FromGeometry<H>,
{
    type Error = GraphError;

    /// Creates a [`MeshGraph`] from a flat [`MeshBuffer`]. The arity of the
    /// polygons in the index buffer must be known and constant.
    ///
    /// # Errors
    ///
    /// Returns an error if a [`MeshGraph`] cannot represent the topology in the
    /// [`MeshBuffer`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use nalgebra::Point2;
    /// use plexus::buffer::MeshBuffer;
    /// use plexus::graph::MeshGraph;
    /// use plexus::index::Flat4;
    /// use plexus::prelude::*;
    /// use std::convert::TryFrom;
    ///
    /// type E2 = Point2<f64>;
    ///
    /// let buffer = MeshBuffer::<Flat4, E2>::from_raw_buffers(
    ///     vec![0u64, 1, 2, 3],
    ///     vec![(0.0f64, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    /// )
    /// .unwrap();
    /// let mut graph = MeshGraph::<E2>::try_from(buffer).unwrap();
    /// ```
    ///
    /// [`MeshBuffer`]: crate::buffer::MeshBuffer
    /// [`MeshGraph`]: crate::graph::MeshGraph
    fn try_from(buffer: MeshBuffer<Flat<A, N>, H>) -> Result<Self, Self::Error> {
        let arity = buffer.arity();
        let (indices, vertices) = buffer.into_raw_buffers();
        MeshGraph::from_raw_buffers_with_arity(indices, vertices, arity)
    }
}

impl<P, H, G> TryFrom<MeshBuffer<P, H>> for MeshGraph<G>
where
    P: Grouping<Group = P> + IntoVertices + Polygonal,
    P::Vertex: Copy + Integer + NumCast + Unsigned,
    H: Clone,
    G: GraphData,
    G::Vertex: FromGeometry<H>,
{
    type Error = GraphError;

    /// Creates a [`MeshGraph`] from a structured [`MeshBuffer`].
    ///
    /// # Errors
    ///
    /// Returns an error if a [`MeshGraph`] cannot represent the topology in the
    /// [`MeshBuffer`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use nalgebra::Point2;
    /// use plexus::buffer::MeshBuffer;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::Tetragon;
    /// use std::convert::TryFrom;
    ///
    /// type E2 = Point2<f64>;
    ///
    /// let buffer = MeshBuffer::<Tetragon<u64>, E2>::from_raw_buffers(
    ///     vec![Tetragon::new(0u64, 1, 2, 3)],
    ///     vec![(0.0f64, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    /// )
    /// .unwrap();
    /// let mut graph = MeshGraph::<E2>::try_from(buffer).unwrap();
    /// ```
    ///
    /// [`MeshBuffer`]: crate::buffer::MeshBuffer
    /// [`MeshGraph`]: crate::graph::MeshGraph
    fn try_from(buffer: MeshBuffer<P, H>) -> Result<Self, Self::Error> {
        let (indices, vertices) = buffer.into_raw_buffers();
        MeshGraph::from_raw_buffers(indices, vertices)
    }
}

#[cfg(test)]
mod tests {
    use decorum::R64;
    use nalgebra::{Point2, Point3, Vector3};
    use num::Zero;

    use crate::buffer::MeshBuffer3;
    use crate::graph::{GraphData, GraphError, MeshGraph};
    use crate::prelude::*;
    use crate::primitive::generate::Position;
    use crate::primitive::sphere::UvSphere;
    use crate::primitive::NGon;

    type E2 = Point2<R64>;
    type E3 = Point3<R64>;

    #[test]
    fn collect() {
        let graph: MeshGraph<Point3<f64>> = UvSphere::new(3, 2)
            .polygons::<Position<E3>>() // 6 triangles, 18 vertices.
            .collect();

        assert_eq!(5, graph.vertex_count());
        assert_eq!(18, graph.arc_count());
        assert_eq!(6, graph.face_count());
    }

    #[test]
    fn iterate() {
        let mut graph: MeshGraph<Point3<f64>> = UvSphere::new(4, 2)
            .polygons::<Position<E3>>() // 8 triangles, 24 vertices.
            .collect();

        assert_eq!(6, graph.vertices().count());
        assert_eq!(24, graph.arcs().count());
        assert_eq!(8, graph.faces().count());
        for vertex in graph.vertices() {
            // Every vertex is connected to 4 triangles with 4 (incoming) arcs.
            // Traversal of topology should be possible.
            assert_eq!(4, vertex.incoming_arcs().count());
        }
        for mut vertex in graph.vertex_orphans() {
            // Data should be mutable.
            *vertex.get_mut() += Vector3::zero();
        }
    }

    #[test]
    fn isolate_disjoint_subgraphs() {
        // Construct a graph from a quadrilateral.
        let graph = MeshGraph::<E2>::from_raw_buffers(
            vec![NGon([0u32, 1, 2, 3])],
            vec![(1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0)],
        )
        .unwrap();

        assert_eq!(1, graph.disjoint_subgraph_vertices().count());

        // Construct a graph with two disjoint quadrilaterals.
        let graph = MeshGraph::<E2>::from_raw_buffers(
            vec![NGon([0u32, 1, 2, 3]), NGon([4, 5, 6, 7])],
            vec![
                (-2.0, 0.0),
                (-1.0, 0.0),
                (-1.0, 1.0),
                (-2.0, 1.0),
                (1.0, 0.0),
                (2.0, 0.0),
                (2.0, 1.0),
                (1.0, 1.0),
            ],
        )
        .unwrap();

        assert_eq!(2, graph.disjoint_subgraph_vertices().count());
    }

    #[test]
    fn non_manifold_error_deferred() {
        let graph: MeshGraph<E3> = UvSphere::new(32, 32)
            .polygons::<Position<E3>>()
            .triangulate()
            .collect();
        // This conversion will join faces by a single vertex, but ultimately
        // creates a manifold.
        let _: MeshBuffer3<usize, E3> = graph.to_mesh_by_face().unwrap();
    }

    #[test]
    fn error_on_non_manifold() {
        // Construct a graph with a "fan" of three triangles sharing the same
        // edge along the Z-axis. The edge would have three associated faces,
        // which should not be possible.
        let graph = MeshGraph::<Point3<i32>>::from_raw_buffers(
            vec![NGon([0u32, 1, 2]), NGon([0, 1, 3]), NGon([0, 1, 4])],
            vec![(0, 0, 1), (0, 0, -1), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
        );

        assert_eq!(graph.err().unwrap(), GraphError::TopologyConflict);
    }

    // This test is a sanity check for iterators over orphan views and the
    // unsafe transmutations used to coerce lifetimes.
    #[test]
    fn read_write_geometry_ref() {
        struct Weight;

        impl GraphData for Weight {
            type Vertex = Point3<f64>;
            type Arc = ();
            type Edge = ();
            type Face = u64;
        }

        // Create a graph with a floating-point weight in each face. Use an
        // iterator over orphan views to write to the geometry of each face.
        let mut graph: MeshGraph<Weight> = UvSphere::new(4, 4).polygons::<Position<E3>>().collect();
        let value = 123_456_789;
        for mut face in graph.face_orphans() {
            *face.get_mut() = value;
        }

        // Read the geometry of each face to ensure it is what we expect.
        for face in graph.faces() {
            assert_eq!(value, face.data);
        }
    }
}
