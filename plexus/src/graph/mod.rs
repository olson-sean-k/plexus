//! Half-edge graph representation of meshes.
//!
//! This module provides a flexible representation of meshes as a [half-edge
//! graph][1]. _Half-edges_ and _edges_ are referred to as _arcs_ and _edges_,
//! respectively. Meshes can store arbitrary geometric data associated with any
//! topological structure (vertices, arcs, edges, and faces).
//!
//! Geometry is vertex-based, meaning that geometric operations depend on
//! vertices exposing some notion of positional data. See the `geometry` module
//! and `AsPosition` trait. If geometry does not have this property, then
//! spatial operations will not be available.
//!
//! See the [user guide][2] for more details and examples.
//!
//! # Representation
//!
//! A `MeshGraph` is conceptually composed of _vertices_, _arcs_, _edges_, and
//! _faces_. The figure below summarizes the connectivity in a `MeshGraph`.
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
//! Arcs are connected to their neighbors, known as _next_ and _previous arcs_.
//! A traversal along a series of arcs is a _path_. The path formed by
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
//! effecient traversals of topology. For example, it becomes trivial to find
//! neighboring topologies, such as the faces that share a given vertex or the
//! neighboring faces of a given face.
//!
//! `MeshGraph`s store topological data using associative collections and mesh
//! data is accessed using keys into this storage. Keys are exposed as strongly
//! typed and opaque values, which can be used to refer to a topological
//! structure.
//!
//! # Topological Views
//!
//! `MeshGraph`s expose _views_ over their topological structures (vertices,
//! arcs, edges, and faces). Views are accessed via keys or iteration and behave
//! similarly to references. They provide the primary API for interacting with a
//! `MeshGraph`'s topology and geometry. There are three types summarized below:
//!
//! | Type      | Traversal | Exclusive | Geometry  | Topology  |
//! |-----------|-----------|-----------|-----------|-----------|
//! | Immutable | Yes       | No        | Immutable | Immutable |
//! | Mutable   | Yes       | Yes       | Mutable   | Mutable   |
//! | Orphan    | No        | No        | Mutable   | N/A       |
//!
//! _Immutable_ and _mutable views_ behave similarly to references: immutable
//! views cannot mutate a graph and are not exclusive while mutable views may
//! mutate both the geometry and topology of a graph but are exclusive.
//!
//! _Orphan views_ are similar to mutable views in that they may mutate the
//! geometry of a graph, but they do not have access to the topology of a graph.
//! Because they do not know about other vertices, arcs, etc., an orphan view
//! cannot traverse a graph in any way. These views are most useful for
//! modifying the geometry of a graph and, unlike mutable views, they are not
//! exclusive. Iterators over topological structures in a graph sometimes emit
//! orphan views.
//!
//! # Examples
//!
//! Generating a graph from a $uv$-sphere:
//!
//! ```rust
//! # extern crate decorum;
//! # extern crate nalgebra;
//! # extern crate plexus;
//! #
//! use decorum::N64;
//! use nalgebra::Point3;
//! use plexus::graph::MeshGraph;
//! use plexus::prelude::*;
//! use plexus::primitive::generate::Position;
//! use plexus::primitive::sphere::UvSphere;
//!
//! let mut graph = UvSphere::default()
//!     .polygons::<Position<Point3<N64>>>()
//!     .collect::<MeshGraph<Point3<N64>>>();
//! ```
//!
//! Extruding a face in a graph:
//!
//! ```rust
//! # extern crate decorum;
//! # extern crate nalgebra;
//! # extern crate plexus;
//! #
//! use decorum::N64;
//! use nalgebra::Point3;
//! use plexus::graph::MeshGraph;
//! use plexus::prelude::*;
//! use plexus::primitive::generate::Position;
//! use plexus::primitive::sphere::UvSphere;
//!
//! let mut graph = UvSphere::new(8, 8)
//!     .polygons::<Position<Point3<N64>>>()
//!     .collect::<MeshGraph<Point3<N64>>>();
//! let key = graph.faces().nth(0).unwrap().key(); // Get the key of the first face.
//! let face = graph.face_mut(key).unwrap().extrude(1.0); // Extrude the face.
//! ```
//!
//! Traversing and circulating over a graph:
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
//! for mut face in vertex.neighboring_face_orphans() {
//!     // `face.geometry` is mutable here.
//! }
//! ```
//!
//! [1]: https://en.wikipedia.org/wiki/doubly_connected_edge_list
//! [2]: https://plexus.rs/user-guide/graphs

mod builder;
mod core;
mod edge;
mod face;
mod geometry;
mod mutation;
mod path;
mod storage;
mod trace;
mod vertex;

use decorum::N64;
use itertools::Itertools;
use num::{Integer, NumCast, ToPrimitive, Unsigned};
use smallvec::SmallVec;
use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::FromIterator;
use theon::ops::Map;
use theon::query::Aabb;
use theon::space::{EuclideanSpace, Scalar};
use theon::AsPosition;
use thiserror::Error;
use typenum::{self, NonZero};

use crate::buffer::{BufferError, FromRawBuffers, FromRawBuffersWithArity, MeshBuffer};
use crate::builder::{Buildable, FacetBuilder, MeshBuilder, SurfaceBuilder};
use crate::encoding::{FaceDecoder, FromEncoding, VertexDecoder};
use crate::graph::builder::GraphBuilder;
use crate::graph::core::{Core, OwnedCore};
use crate::graph::geometry::Geometric;
use crate::graph::mutation::face::FaceInsertCache;
use crate::graph::mutation::{Consistent, Mutation};
use crate::graph::storage::*;
use crate::index::{Flat, FromIndexer, Grouping, HashIndexer, IndexBuffer, IndexVertices, Indexer};
use crate::network::storage::{AsStorage, AsStorageMut, Fuse, OpaqueKey, Storage};
use crate::network::view::{Bind, Orphan, View};
use crate::primitive::decompose::IntoVertices;
use crate::primitive::Polygonal;
use crate::transact::Transact;
use crate::{DynamicArity, FromGeometry, IntoGeometry, MeshArity, StaticArity};

pub use crate::graph::edge::{
    Arc, ArcKey, ArcOrphan, ArcView, Edge, EdgeKey, EdgeOrphan, EdgeView, Edgoid,
};
pub use crate::graph::face::{Face, FaceKey, FaceOrphan, FaceView, Ring, Ringoid};
pub use crate::graph::geometry::{
    ArcNormal, EdgeMidpoint, FaceCentroid, FaceNormal, FacePlane, GraphGeometry, VertexCentroid,
    VertexNormal, VertexPosition,
};
pub use crate::graph::path::Path;
pub use crate::graph::vertex::{Vertex, VertexKey, VertexOrphan, VertexView};
pub use crate::network::view::{ClosedView, Rebind};

pub use Selector::ByIndex;
pub use Selector::ByKey;

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
    #[error("conflicting arity; expected {expected}, but got {actual}")]
    ArityConflict { expected: usize, actual: usize },
    #[error("arity is non-uniform")]
    ArityNonUniform,
    #[error("geometric operation failed")]
    Geometry,
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

/// Topology selector.
///
/// Identifies topology by key or index. Keys behave as an absolute selector and
/// uniquely identify a single topological structure. Indices behave as a
/// relative selector and identify topological structures relative to some other
/// structure. `Selector` is used by operations that support both of these
/// selection mechanisms.
///
/// An index is typically used to select a neighbor or contained (and ordered)
/// topological structure, such as a neighboring face.
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
/// use decorum::N64;
/// use nalgebra::Point3;
/// use plexus::graph::MeshGraph;
/// use plexus::prelude::*;
/// use plexus::primitive::cube::Cube;
/// use plexus::primitive::generate::Position;
///
/// let mut graph = Cube::new()
///     .polygons::<Position<Point3<N64>>>()
///     .collect::<MeshGraph<Point3<N64>>>();
/// let key = graph.faces().nth(0).unwrap().key();
/// graph
///     .face_mut(key)
///     .unwrap()
///     .split(ByIndex(0), ByIndex(2))
///     .unwrap();
/// ```
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
    K: OpaqueKey,
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

/// Half-edge graph representation of a mesh.
///
/// Provides topological data in the form of vertices, arcs, edges, and faces.
/// An arc is directed from one vertex to another, with an opposing arc joining
/// the vertices in the other direction.
///
/// `MeshGraph`s expose topological views, which can be used to traverse and
/// manipulate topology and geometry in the graph.
///
/// See the module documentation for more details.
pub struct MeshGraph<G = (N64, N64, N64)>
where
    G: GraphGeometry,
{
    core: OwnedCore<G>,
}

impl<G> MeshGraph<G>
where
    G: GraphGeometry,
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
                .fuse(Storage::<Vertex<G>>::new())
                .fuse(Storage::<Arc<G>>::new())
                .fuse(Storage::<Edge<G>>::new())
                .fuse(Storage::<Face<G>>::new()),
        )
    }

    /// Gets the number of vertices in the graph.
    pub fn vertex_count(&self) -> usize {
        self.as_vertex_storage().len()
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
    pub fn vertices(&self) -> impl ExactSizeIterator<Item = VertexView<&Self>> {
        self.as_vertex_storage()
            .keys()
            .map(move |key| View::bind_unchecked(self, key))
            .map(From::from)
    }

    /// Gets an iterator of orphan views over the vertices in the graph.
    ///
    /// Because this only yields orphan views, only geometry can be mutated.
    /// For topological mutations, collect the necessary keys and use
    /// `vertex_mut` instead.
    pub fn vertex_orphans(&mut self) -> impl ExactSizeIterator<Item = VertexOrphan<G>> {
        self.as_vertex_storage_mut()
            .iter_mut()
            .map(|(key, entity)| Orphan::bind_unchecked(entity, key))
            .map(From::from)
    }

    /// Gets the number of arcs in the graph.
    pub fn arc_count(&self) -> usize {
        self.as_arc_storage().len()
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
    pub fn arcs(&self) -> impl ExactSizeIterator<Item = ArcView<&Self>> {
        self.as_arc_storage()
            .keys()
            .map(move |key| View::bind_unchecked(self, key))
            .map(From::from)
    }

    /// Gets an iterator of orphan views over the arcs in the graph.
    ///
    /// Because this only yields orphan views, only geometry can be mutated.
    /// For topological mutations, collect the necessary keys and use `arc_mut`
    /// instead.
    pub fn arc_orphans(&mut self) -> impl ExactSizeIterator<Item = ArcOrphan<G>> {
        self.as_arc_storage_mut()
            .iter_mut()
            .map(|(key, entity)| Orphan::bind_unchecked(entity, key))
            .map(From::from)
    }

    /// Gets the number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.as_edge_storage().len()
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
    pub fn edges(&self) -> impl ExactSizeIterator<Item = EdgeView<&Self>> {
        self.as_edge_storage()
            .keys()
            .map(move |key| View::bind_unchecked(self, key))
            .map(From::from)
    }

    /// Gets an iterator of orphan views over the edges in the graph.
    ///
    /// Because this only yields orphan views, only geometry can be mutated.
    /// For topological mutations, collect the necessary keys and use `edge_mut`
    /// instead.
    pub fn edge_orphans(&mut self) -> impl ExactSizeIterator<Item = EdgeOrphan<G>> {
        self.as_edge_storage_mut()
            .iter_mut()
            .map(|(key, entity)| Orphan::bind_unchecked(entity, key))
            .map(From::from)
    }

    /// Gets the number of faces in the graph.
    pub fn face_count(&self) -> usize {
        self.as_face_storage().len()
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
    pub fn faces(&self) -> impl ExactSizeIterator<Item = FaceView<&Self>> {
        self.as_face_storage()
            .keys()
            .map(move |key| View::bind_unchecked(self, key))
            .map(From::from)
    }

    /// Gets an iterator of orphan views over the faces in the graph.
    ///
    /// Because this only yields orphan views, only geometry can be mutated.
    /// For topological mutations, collect the necessary keys and use `face_mut`
    /// instead.
    pub fn face_orphans(&mut self) -> impl ExactSizeIterator<Item = FaceOrphan<G>> {
        self.as_face_storage_mut()
            .iter_mut()
            .map(|(key, entity)| Orphan::bind_unchecked(entity, key))
            .map(From::from)
    }

    pub fn path<I>(&self, keys: I) -> Result<Path<&Self>, GraphError>
    where
        I: IntoIterator,
        I::Item: Borrow<VertexKey>,
    {
        Path::bind(self, keys)
    }

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
    {
        Aabb::from_points(self.vertices().map(|vertex| *vertex.geometry.as_position()))
    }

    /// Triangulates the graph, tessellating all faces into triangles.
    pub fn triangulate(&mut self) {
        let faces = self.as_face_storage().keys().collect::<Vec<_>>();
        for face in faces {
            self.face_mut(face).unwrap().triangulate();
        }
    }

    /// Smooths the positions of vertices in the graph.
    ///
    /// Each position is translated by its offset from its centroid scaled by
    /// the given factor. The centroid of a vertex position is the mean of the
    /// positions of its neighboring vertices. That is, given a factor $k$ and a
    /// vertex with position $P$ and centroid $Q$, its position becomes
    /// $P+k(Q-P)$.
    pub fn smooth<T>(&mut self, factor: T)
    where
        T: Into<Scalar<VertexPosition<G>>>,
        G: VertexCentroid,
        G::Vertex: AsPosition,
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
            *vertex.geometry.as_position_mut() = positions.remove(&vertex.key()).unwrap();
        }
    }

    // This API is a bit unusual, but allows a view-like path to borrow a graph
    // and remain consistent. It also (hopefully) makes it more clear that the
    // _graph_ is split, not the path.
    //
    //   let mut path = graph.arc_mut(...).unwrap().into_path();
    //   path.push(...).unwrap();
    //   ...
    //   let (a, b) = MeshGraph::split_at_path(path).unwrap();
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

    /// Gets an iterator over a vertex within each disjoint subgraph.
    ///
    /// Traverses the graph and returns an arbitrary vertex within each
    /// _disjoint subgraph_. A subgraph is _disjoint_ if it cannot be reached
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
        let keys = self.as_vertex_storage().keys().collect::<HashSet<_>>();
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

    /// Creates a `Buildable` mesh data structure from the graph.
    ///
    /// The output is created from each unique vertex in the graph. No face
    /// geometry is used, and the `Facet` type is always `()`.
    ///
    /// # Examples
    ///
    /// Creating a `MeshBuffer` from a graph used to modify a cube:
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
    /// let mut graph = Cube::new()
    ///     .polygons::<Position<E3>>()
    ///     .collect::<MeshGraph<E3>>();
    /// let key = graph.faces().nth(0).unwrap().key();
    /// graph.face_mut(key).unwrap().extrude(1.0);
    ///
    /// let buffer = graph.to_mesh_by_vertex::<MeshBufferN<usize, E3>>().unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the graph does not have constant arity that is
    /// compatible with the index buffer. Typically, a graph is triangulated
    /// before being converted to a buffer.
    pub fn to_mesh_by_vertex<B>(&self) -> Result<B, B::Error>
    where
        B: Buildable<Facet = ()>,
        B::Vertex: FromGeometry<G::Vertex>,
    {
        self.to_mesh_by_vertex_with(|vertex| vertex.geometry.into_geometry())
    }

    /// Creates a `Buildable` mesh data structure from the graph.
    ///
    /// The output is created from each unique vertex in the graph, which is
    /// converted into the output geometry by the given function. No face
    /// geometry is used, and the `Facet` type is always `()`.
    ///
    /// # Errors
    ///
    /// Returns an error if the vertex geometry cannot be inserted into the
    /// output, there are arity conflicts, or the output does not support
    /// topology found in the graph.
    pub fn to_mesh_by_vertex_with<B, T, F>(&self, mut f: F) -> Result<B, B::Error>
    where
        B: Buildable<Vertex = T, Facet = ()>,
        F: FnMut(VertexView<&Self>) -> T,
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
                        .vertices()
                        .map(|vertex| keys[&vertex.key()])
                        .collect::<SmallVec<[_; 8]>>();
                    builder.insert_facet(indices.as_slice(), ())?;
                }
                Ok(())
            })
        })?;
        builder.build()
    }

    /// Creates a `Buildable` mesh data structure from the graph.
    ///
    /// The output is created from each face in the graph. For each face, the
    /// face geometry and associated vertex geometry is inserted into the buffer
    /// via `FromGeometry`. This means that a vertex is inserted for each of its
    /// adjacent faces.
    ///
    /// # Errors
    ///
    /// Returns an error if the vertex geometry cannot be inserted into the
    /// output, there are arity conflicts, or the output does not support
    /// topology found in the graph.
    pub fn to_mesh_by_face<B>(&self) -> Result<B, B::Error>
    where
        B: Buildable,
        B::Vertex: FromGeometry<G::Vertex>,
        B::Facet: FromGeometry<G::Face>,
    {
        self.to_mesh_by_face_with(|_, vertex| vertex.geometry.into_geometry())
    }

    /// Creates a `Buildable` mesh data structure from the graph.
    ///
    /// The output is created from each face in the graph. The given function is
    /// called for each vertex of each face and converts the vertex geometry
    /// into the output geometry. This means that a vertex is inserted for each
    /// of its adjacent faces. The face geometry is inserted into the output via
    /// `FromGeometry`.
    ///
    /// # Examples
    ///
    /// Creating a `MeshBuffer` from a graph loaded from PLY data and used to
    /// compute normals:
    ///
    /// ```rust
    /// # extern crate decorum;
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// # extern crate theon;
    /// #
    /// use decorum::N64;
    /// use nalgebra::Point3;
    /// use plexus::buffer::MeshBuffer;
    /// use plexus::encoding::ply::{FromPly, PositionEncoding};
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::Polygon;
    /// use theon::space::Vector;
    ///
    /// type E3 = Point3<N64>;
    ///
    /// pub struct Vertex {
    ///     pub position: E3,
    ///     pub normal: Vector<E3>,
    /// }
    ///
    /// let ply: &[u8] = {
    ///     // ...
    /// #     include_bytes!("../../../data/cube.ply")
    /// };
    /// let encoding = PositionEncoding::<E3>::default();
    /// let (graph, _) = MeshGraph::<E3>::from_ply(encoding, ply).unwrap();
    ///
    /// let buffer: MeshBuffer<Polygon<usize>, _> = graph
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
    pub fn to_mesh_by_face_with<B, T, F>(&self, mut f: F) -> Result<B, B::Error>
    where
        B: Buildable<Vertex = T>,
        B::Facet: FromGeometry<G::Face>,
        F: FnMut(FaceView<&Self>, VertexView<&Self>) -> T,
    {
        let mut builder = B::builder();
        builder.surface_with(|builder| {
            for face in self.faces() {
                let indices = face
                    .vertices()
                    .map(|vertex| builder.insert_vertex(f(face, vertex)))
                    .collect::<Result<SmallVec<[_; 8]>, _>>()?;
                builder.facets_with(|builder| {
                    builder.insert_facet(indices.as_slice(), face.geometry)
                })?;
            }
            Ok(())
        })?;
        builder.build()
    }
}

impl<G> AsStorage<Vertex<G>> for MeshGraph<G>
where
    G: GraphGeometry,
{
    fn as_storage(&self) -> &Storage<Vertex<G>> {
        self.core.as_vertex_storage()
    }
}

impl<G> AsStorage<Arc<G>> for MeshGraph<G>
where
    G: GraphGeometry,
{
    fn as_storage(&self) -> &Storage<Arc<G>> {
        self.core.as_arc_storage()
    }
}

impl<G> AsStorage<Edge<G>> for MeshGraph<G>
where
    G: GraphGeometry,
{
    fn as_storage(&self) -> &Storage<Edge<G>> {
        self.core.as_edge_storage()
    }
}

impl<G> AsStorage<Face<G>> for MeshGraph<G>
where
    G: GraphGeometry,
{
    fn as_storage(&self) -> &Storage<Face<G>> {
        self.core.as_face_storage()
    }
}

impl<G> AsStorageMut<Vertex<G>> for MeshGraph<G>
where
    G: GraphGeometry,
{
    fn as_storage_mut(&mut self) -> &mut Storage<Vertex<G>> {
        self.core.as_vertex_storage_mut()
    }
}

impl<G> AsStorageMut<Arc<G>> for MeshGraph<G>
where
    G: GraphGeometry,
{
    fn as_storage_mut(&mut self) -> &mut Storage<Arc<G>> {
        self.core.as_arc_storage_mut()
    }
}

impl<G> AsStorageMut<Edge<G>> for MeshGraph<G>
where
    G: GraphGeometry,
{
    fn as_storage_mut(&mut self) -> &mut Storage<Edge<G>> {
        self.core.as_edge_storage_mut()
    }
}

impl<G> AsStorageMut<Face<G>> for MeshGraph<G>
where
    G: GraphGeometry,
{
    fn as_storage_mut(&mut self) -> &mut Storage<Face<G>> {
        self.core.as_face_storage_mut()
    }
}

/// Exposes a `MeshBuilder` that can be used to construct a `MeshGraph`
/// incrementally from _surfaces_ and _facets_.
///
/// See the documentation for the `builder` module for more.
///
/// # Examples
///
/// Creating a graph from a triangle:
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
impl<G> Buildable for MeshGraph<G>
where
    G: GraphGeometry,
{
    type Builder = GraphBuilder<G>;
    type Error = GraphError;

    type Vertex = G::Vertex;
    type Facet = G::Face;

    fn builder() -> Self::Builder {
        Default::default()
    }
}

impl<G> Consistent for MeshGraph<G> where G: GraphGeometry {}

impl<G> Default for MeshGraph<G>
where
    G: GraphGeometry,
{
    fn default() -> Self {
        MeshGraph::new()
    }
}

impl<G> DynamicArity for MeshGraph<G>
where
    G: GraphGeometry,
{
    type Dynamic = MeshArity;

    fn arity(&self) -> Self::Dynamic {
        MeshArity::from_components(self.faces())
    }
}

impl<G> From<OwnedCore<G>> for MeshGraph<G>
where
    G: GraphGeometry,
{
    fn from(core: OwnedCore<G>) -> Self {
        MeshGraph { core }
    }
}

impl<E, G> FromEncoding<E> for MeshGraph<G>
where
    G: GraphGeometry,
    E: FaceDecoder + VertexDecoder,
    E::Face: IntoGeometry<G::Face>,
    E::Vertex: IntoGeometry<G::Vertex>,
{
    type Error = GraphError;

    fn from_encoding(
        vertices: <E as VertexDecoder>::Output,
        faces: <E as FaceDecoder>::Output,
    ) -> Result<Self, Self::Error> {
        let mut mutation = Mutation::from(MeshGraph::new());
        let keys = vertices
            .into_iter()
            .map(|geometry| mutation.insert_vertex(geometry.into_geometry()))
            .collect::<Vec<_>>();
        for (perimeter, geometry) in faces {
            let perimeter = perimeter
                .into_iter()
                .map(|index| keys[index])
                .collect::<SmallVec<[_; 4]>>();
            let cache = FaceInsertCache::snapshot(&mutation, perimeter.as_slice())?;
            let geometry = geometry.into_geometry();
            mutation.insert_face_with(cache, || (Default::default(), geometry))?;
        }
        mutation.commit()
    }
}

impl<G, P> FromIndexer<P, P> for MeshGraph<G>
where
    G: GraphGeometry,
    P: Map<usize> + Polygonal,
    P::Output: Grouping<Item = P::Output> + IntoVertices + Polygonal<Vertex = usize>,
    P::Vertex: IntoGeometry<G::Vertex>,
    Vec<P::Output>: IndexBuffer<P::Output, Index = usize>,
{
    type Error = GraphError;

    fn from_indexer<I, N>(input: I, indexer: N) -> Result<Self, Self::Error>
    where
        I: IntoIterator<Item = P>,
        N: Indexer<P, P::Vertex>,
    {
        let mut mutation = Mutation::from(MeshGraph::new());
        let (indices, vertices) = input.into_iter().index_vertices(indexer);
        let vertices = vertices
            .into_iter()
            .map(|vertex| mutation.insert_vertex(vertex.into_geometry()))
            .collect::<Vec<_>>();
        for face in indices {
            let perimeter = face
                .into_vertices()
                .into_iter()
                .map(|index| vertices[index])
                .collect::<SmallVec<[_; 4]>>();
            let cache = FaceInsertCache::snapshot(&mutation, &perimeter)?;
            mutation.insert_face_with(cache, Default::default)?;
        }
        mutation.commit()
    }
}

impl<G, P> FromIterator<P> for MeshGraph<G>
where
    G: GraphGeometry,
    P: Polygonal,
    P::Vertex: Clone + Eq + Hash + IntoGeometry<G::Vertex>,
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
    G: GraphGeometry,
    H: IntoGeometry<G::Vertex>,
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
            .map(|vertex| mutation.insert_vertex(vertex.into_geometry()))
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
            let cache = FaceInsertCache::snapshot(&mutation, &perimeter)?;
            mutation.insert_face_with(cache, Default::default)?;
        }
        mutation.commit()
    }
}

impl<N, G, H> FromRawBuffersWithArity<N, H> for MeshGraph<G>
where
    N: Integer + ToPrimitive + Unsigned,
    G: GraphGeometry,
    H: IntoGeometry<G::Vertex>,
{
    type Error = GraphError;

    /// Creates a `MeshGraph` from raw index and vertex buffers. The arity of
    /// the polygons in the index buffer must be known and constant.
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
    /// let (indices, positions) = UvSphere::new(16, 16)
    ///     .polygons::<Position<Point3<f64>>>()
    ///     .triangulate()
    ///     .index_vertices::<Flat3, _>(LruIndexer::with_capacity(256));
    /// let mut graph =
    ///     MeshGraph::<Point3<f64>>::from_raw_buffers_with_arity(indices, positions, 3).unwrap();
    /// ```
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
            .map(|vertex| mutation.insert_vertex(vertex.into_geometry()))
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
            let cache = FaceInsertCache::snapshot(&mutation, &perimeter)?;
            mutation.insert_face_with(cache, Default::default)?;
        }
        mutation.commit()
    }
}

impl<G> Geometric for MeshGraph<G>
where
    G: GraphGeometry,
{
    type Geometry = G;
}

impl<G> Into<OwnedCore<G>> for MeshGraph<G>
where
    G: GraphGeometry,
{
    fn into(self) -> OwnedCore<G> {
        let MeshGraph { core, .. } = self;
        core
    }
}

impl<G> StaticArity for MeshGraph<G>
where
    G: GraphGeometry,
{
    type Static = (usize, Option<usize>);

    const ARITY: Self::Static = (3, None);
}

impl<A, N, H, G> TryFrom<MeshBuffer<Flat<A, N>, H>> for MeshGraph<G>
where
    A: NonZero + typenum::Unsigned,
    N: Copy + Integer + NumCast + Unsigned,
    H: Clone + IntoGeometry<G::Vertex>,
    G: GraphGeometry,
{
    type Error = GraphError;

    /// Creates a `MeshGraph` from a flat `MeshBuffer`. The arity of the
    /// polygons in the index buffer must be known and constant.
    ///
    /// # Errors
    ///
    /// Returns an error if a `MeshGraph` cannot represent the topology in the
    /// `MeshBuffer`.
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
    /// let buffer = MeshBuffer::<Flat4, _>::from_raw_buffers(
    ///     vec![0u64, 1, 2, 3],
    ///     vec![(0.0f64, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    /// )
    /// .unwrap();
    /// let mut graph = MeshGraph::<Point2<f64>>::try_from(buffer).unwrap();
    /// ```
    fn try_from(buffer: MeshBuffer<Flat<A, N>, H>) -> Result<Self, Self::Error> {
        let arity = buffer.arity();
        let (indices, vertices) = buffer.into_raw_buffers();
        MeshGraph::from_raw_buffers_with_arity(indices, vertices, arity)
    }
}

impl<P, H, G> TryFrom<MeshBuffer<P, H>> for MeshGraph<G>
where
    P: Grouping<Item = P> + IntoVertices + Polygonal,
    P::Vertex: Copy + Integer + NumCast + Unsigned,
    H: Clone + IntoGeometry<G::Vertex>,
    G: GraphGeometry,
{
    type Error = GraphError;

    /// Creates a `MeshGraph` from a structured `MeshBuffer`.
    ///
    /// # Errors
    ///
    /// Returns an error if a `MeshGraph` cannot represent the topology in the
    /// `MeshBuffer`.
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
    /// let buffer = MeshBuffer::<Tetragon<u64>, _>::from_raw_buffers(
    ///     vec![Tetragon::new(0u64, 1, 2, 3)],
    ///     vec![(0.0f64, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    /// )
    /// .unwrap();
    /// let mut graph = MeshGraph::<Point2<f64>>::try_from(buffer).unwrap();
    /// ```
    fn try_from(buffer: MeshBuffer<P, H>) -> Result<Self, Self::Error> {
        let (indices, vertices) = buffer.into_raw_buffers();
        MeshGraph::from_raw_buffers(indices, vertices)
    }
}

#[cfg(test)]
mod tests {
    use decorum::N64;
    use nalgebra::{Point2, Point3, Vector3};
    use num::Zero;

    use crate::buffer::MeshBuffer3;
    use crate::graph::{GraphError, GraphGeometry, MeshGraph};
    use crate::prelude::*;
    use crate::primitive::generate::Position;
    use crate::primitive::sphere::UvSphere;
    use crate::primitive::NGon;

    type E2 = Point2<N64>;
    type E3 = Point3<N64>;

    #[test]
    fn collect_topology_into_mesh() {
        let graph = UvSphere::new(3, 2)
            .polygons::<Position<E3>>() // 6 triangles, 18 vertices.
            .collect::<MeshGraph<Point3<f64>>>();

        assert_eq!(5, graph.vertex_count());
        assert_eq!(18, graph.arc_count());
        assert_eq!(6, graph.face_count());
    }

    #[test]
    fn iterate_mesh_topology() {
        let mut graph = UvSphere::new(4, 2)
            .polygons::<Position<E3>>() // 8 triangles, 24 vertices.
            .collect::<MeshGraph<Point3<f64>>>();

        assert_eq!(6, graph.vertices().count());
        assert_eq!(24, graph.arcs().count());
        assert_eq!(8, graph.faces().count());
        for vertex in graph.vertices() {
            // Every vertex is connected to 4 triangles with 4 (incoming) arcs.
            // Traversal of topology should be possible.
            assert_eq!(4, vertex.incoming_arcs().count());
        }
        for mut vertex in graph.vertex_orphans() {
            // Geometry should be mutable.
            vertex.geometry += Vector3::zero();
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
        let graph = UvSphere::new(32, 32)
            .polygons::<Position<E3>>()
            .triangulate()
            .collect::<MeshGraph<E3>>();
        // This conversion will join faces by a single vertex, but ultimately
        // creates a manifold.
        let _: MeshBuffer3<usize, E3> = graph.to_mesh_by_face().unwrap();
    }

    #[test]
    fn error_on_non_manifold_mesh() {
        // Construct a graph with a "fan" of three triangles sharing the same
        // arc along the Z-axis. The edge would have three associated faces,
        // which should not be possible.
        let graph = MeshGraph::<Point3<i32>>::from_raw_buffers(
            vec![NGon([0u32, 1, 2]), NGon([0, 1, 3]), NGon([0, 1, 4])],
            vec![(0, 0, 1), (0, 0, -1), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
        );

        assert_eq!(graph.err().unwrap(), GraphError::TopologyConflict);
    }

    // This test is a sanity check for graph iterators, topological views, and
    // the unsafe transmutations used to coerce lifetimes.
    #[test]
    fn read_write_geometry_ref() {
        struct ValueGeometry;

        impl GraphGeometry for ValueGeometry {
            type Vertex = Point3<f64>;
            type Arc = ();
            type Edge = ();
            type Face = u64;
        }

        // Create a graph with a floating point value associated with each face.
        // Use a mutable iterator to write to the geometry of each face.
        let mut graph = UvSphere::new(4, 4)
            .polygons::<Position<E3>>()
            .collect::<MeshGraph<ValueGeometry>>();
        let value = 123_456_789;
        for mut face in graph.face_orphans() {
            face.geometry = value;
        }

        // Read the geometry of each face using an immutable iterator to ensure
        // it is what we expect.
        for face in graph.faces() {
            assert_eq!(value, face.geometry);
        }
    }
}
