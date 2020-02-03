//! Graph payloads.
//!
//! A _payload_ is data contained in graph storage (via `StorageProxy` and
//! related types). For graphs, payloads are the most basic types that represent
//! a graph, including its connectivity and user geometry.
//!
//! Views are built atop these types to provide the graph API. A limited subset
//! of fields from these types are exposed via `Deref` implementations in views.
//! Most notably, user geometry is exposed via `geometry` fields.
//!
//! Connectivity fields are sometimes `Option` types, but may be required in a
//! consistent graph. These fields are only `Option` types because payloads may
//! have dependencies that cannot be realized when they are initialized.

use derivative::Derivative;

use crate::graph::geometry::GraphGeometry;
use crate::graph::storage::key::{ArcKey, EdgeKey, FaceKey, OpaqueKey, VertexKey};
use crate::graph::storage::{Get, HashStorage, Remove, Sequence, SlotStorage};

/// A payload contained in graph storage.
pub trait Payload: Copy + Sized {
    type Key: OpaqueKey;
    type Attribute: Clone;

    // This associated type allows `Payload`s to specify what underlying
    // type is used to support their `StorageProxy` implementation. This
    // greatly reduces trait complexity for `StorageProxy` and `AsStorage`.
    type Storage: Default + Get<Self> + Remove<Self> + Sequence<Self>;
}

/// Graph vertex.
///
/// A vertex is represented by a key into its leading arc.
#[derivative(Clone, Copy, Debug, Hash)]
#[derive(Derivative)]
pub struct Vertex<G>
where
    G: GraphGeometry,
{
    /// User geometry.
    ///
    /// The type of this field is derived from `GraphGeometry`.
    #[derivative(Debug = "ignore", Hash = "ignore")]
    pub geometry: G::Vertex,
    /// Required key into the leading arc.
    pub(in crate::graph) arc: Option<ArcKey>,
}

impl<G> Vertex<G>
where
    G: GraphGeometry,
{
    pub(in crate::graph) fn new(geometry: G::Vertex) -> Self {
        Vertex {
            geometry,
            arc: None,
        }
    }
}

impl<G> Payload for Vertex<G>
where
    G: GraphGeometry,
{
    type Key = VertexKey;
    type Attribute = G::Vertex;
    type Storage = SlotStorage<Self>;
}

// Unlike other graph structures, the vertex connectivity of an arc is immutable
// and encoded within its key. This provides fast and reliable lookups even when
// a graph is in an inconsistent state. However, it also complicates certain
// topological mutations and sometimes requires that arcs be rekeyed. For this
// reason, `Arc` has no fields representing its source and destination vertices
// nor its opposite arc; such fields would be redundant.
/// Graph arc.
///
/// An arc is represented by keys into its next and previous arcs, a key into
/// its edge, and an optional key into its face (for which is may be a leading
/// arc). Additional information is encoded in arc keys.
#[derivative(Clone, Copy, Debug, Hash)]
#[derive(Derivative)]
pub struct Arc<G>
where
    G: GraphGeometry,
{
    /// User geometry.
    ///
    /// The type of this field is derived from `GraphGeometry`.
    #[derivative(Debug = "ignore", Hash = "ignore")]
    pub geometry: G::Arc,
    /// Required key into the next arc.
    pub(in crate::graph) next: Option<ArcKey>,
    /// Required key into the previous arc.
    pub(in crate::graph) previous: Option<ArcKey>,
    /// Required key into the edge.
    pub(in crate::graph) edge: Option<EdgeKey>,
    /// Optional key into the face.
    pub(in crate::graph) face: Option<FaceKey>,
}

impl<G> Arc<G>
where
    G: GraphGeometry,
{
    pub(in crate::graph) fn new(geometry: G::Arc) -> Self {
        Arc {
            geometry,
            next: None,
            previous: None,
            edge: None,
            face: None,
        }
    }
}

impl<G> Payload for Arc<G>
where
    G: GraphGeometry,
{
    type Key = ArcKey;
    type Attribute = G::Arc;
    type Storage = HashStorage<Self>;
}

/// Graph edge.
///
/// An edge is represented by a key into its leading arc.
#[derivative(Clone, Copy, Debug, Hash)]
#[derive(Derivative)]
pub struct Edge<G>
where
    G: GraphGeometry,
{
    /// User geometry.
    ///
    /// The type of this field is derived from `GraphGeometry`.
    #[derivative(Debug = "ignore", Hash = "ignore")]
    pub geometry: G::Edge,
    /// Required key into the leading arc.
    pub(in crate::graph) arc: ArcKey,
}

impl<G> Edge<G>
where
    G: GraphGeometry,
{
    pub(in crate::graph) fn new(arc: ArcKey, geometry: G::Edge) -> Self {
        Edge { geometry, arc }
    }
}

impl<G> Payload for Edge<G>
where
    G: GraphGeometry,
{
    type Key = EdgeKey;
    type Attribute = G::Edge;
    type Storage = SlotStorage<Self>;
}

/// Graph face.
///
/// A face is represented by a key into its leading arc.
#[derivative(Clone, Copy, Debug, Hash)]
#[derive(Derivative)]
pub struct Face<G>
where
    G: GraphGeometry,
{
    /// User geometry.
    ///
    /// The type of this field is derived from `GraphGeometry`.
    #[derivative(Debug = "ignore", Hash = "ignore")]
    pub geometry: G::Face,
    /// Required key into the leading arc.
    pub(in crate::graph) arc: ArcKey,
}

impl<G> Face<G>
where
    G: GraphGeometry,
{
    pub(in crate::graph) fn new(arc: ArcKey, geometry: G::Face) -> Self {
        Face { geometry, arc }
    }
}

impl<G> Payload for Face<G>
where
    G: GraphGeometry,
{
    type Key = FaceKey;
    type Attribute = G::Face;
    type Storage = SlotStorage<Self>;
}
