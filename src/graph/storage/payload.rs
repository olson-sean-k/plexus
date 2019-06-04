//! Graph payloads.
//!
//! This module provides types that store the user geometry and connectivity of
//! graphs. A limited subset of fields from these types are exposed via `Deref`
//! implementations in views. Most notably, user geometry is exposed via the
//! `geometry` field.
//!
//! These types are stored using the `storage` module and `StorageProxy`.

use crate::geometry::{FromGeometry, FromInteriorGeometry, IntoGeometry};
use crate::graph::geometry::GraphGeometry;
use crate::graph::storage::key::{ArcKey, EdgeKey, FaceKey, OpaqueKey, VertexKey};
use crate::graph::storage::{Get, HashStorage, Remove, Sequence, SlotStorage};

pub trait Payload: Copy + Sized {
    type Key: OpaqueKey;
    type Attribute: Clone;

    // This associated type allows `Payload`s to specify what underlying
    // type is used to support their `StorageProxy` implementation. This
    // greatly reduces trait complexity for `StorageProxy` and `AsStorage`.
    type Storage: Default + Get<Self> + Remove<Self> + Sequence<Self>;
}

/// Vertex payload.
///
/// Contains the vertex attribute of `GraphGeometry`.
#[derivative(Clone, Copy, Debug, Hash)]
#[derive(Derivative)]
pub struct VertexPayload<G>
where
    G: GraphGeometry,
{
    #[derivative(Debug = "ignore", Hash = "ignore")]
    pub geometry: G::Vertex,
    pub(in crate::graph) arc: Option<ArcKey>,
}

impl<G> VertexPayload<G>
where
    G: GraphGeometry,
{
    pub(in crate::graph) fn new(geometry: G::Vertex) -> Self {
        VertexPayload {
            geometry,
            arc: None,
        }
    }
}

impl<G, H> FromInteriorGeometry<VertexPayload<H>> for VertexPayload<G>
where
    G: GraphGeometry,
    G::Vertex: FromGeometry<H::Vertex>,
    H: GraphGeometry,
{
    fn from_interior_geometry(vertex: VertexPayload<H>) -> Self {
        VertexPayload {
            geometry: vertex.geometry.into_geometry(),
            arc: vertex.arc,
        }
    }
}

impl<G> Payload for VertexPayload<G>
where
    G: GraphGeometry,
{
    type Key = VertexKey;
    type Attribute = G::Vertex;
    type Storage = SlotStorage<Self>;
}

// Unlike other topological structures, the vertex connectivity of arcs is
// immutable and encoded within the key for each arc. An arc key consists of
// its source and destination vertex keys. This provides fast and reliable arc
// lookups, even when a mesh is in an inconsistent state. However, it also
// complicates basic mutations of vertices and arcs, requiring arcs to be
// rekeyed.
//
// For this reason, `ArcPayload` has no fields for storing its destination
// vertex key or opposite arc key, as these would be redundant.
/// Arc payload.
///
/// Contains the arc attribute of `GraphGeometry`.
#[derivative(Clone, Copy, Debug, Hash)]
#[derive(Derivative)]
pub struct ArcPayload<G>
where
    G: GraphGeometry,
{
    #[derivative(Debug = "ignore", Hash = "ignore")]
    pub geometry: G::Arc,
    pub(in crate::graph) next: Option<ArcKey>,
    pub(in crate::graph) previous: Option<ArcKey>,
    pub(in crate::graph) edge: Option<EdgeKey>,
    pub(in crate::graph) face: Option<FaceKey>,
}

impl<G> ArcPayload<G>
where
    G: GraphGeometry,
{
    pub(in crate::graph) fn new(geometry: G::Arc) -> Self {
        ArcPayload {
            geometry,
            next: None,
            previous: None,
            edge: None,
            face: None,
        }
    }
}

impl<G, H> FromInteriorGeometry<ArcPayload<H>> for ArcPayload<G>
where
    G: GraphGeometry,
    G::Arc: FromGeometry<H::Arc>,
    H: GraphGeometry,
{
    fn from_interior_geometry(arc: ArcPayload<H>) -> Self {
        ArcPayload {
            geometry: arc.geometry.into_geometry(),
            next: arc.next,
            previous: arc.previous,
            edge: arc.edge,
            face: arc.face,
        }
    }
}

impl<G> Payload for ArcPayload<G>
where
    G: GraphGeometry,
{
    type Key = ArcKey;
    type Attribute = G::Arc;
    type Storage = HashStorage<Self>;
}

/// Edge payload.
///
/// Contains the edge attribute of `GraphGeometry`.
#[derivative(Clone, Copy, Debug, Hash)]
#[derive(Derivative)]
pub struct EdgePayload<G>
where
    G: GraphGeometry,
{
    #[derivative(Debug = "ignore", Hash = "ignore")]
    pub geometry: G::Edge,
    pub(in crate::graph) arc: ArcKey,
}

impl<G> EdgePayload<G>
where
    G: GraphGeometry,
{
    pub(in crate::graph) fn new(arc: ArcKey, geometry: G::Edge) -> Self {
        EdgePayload { geometry, arc }
    }
}

impl<G, H> FromInteriorGeometry<EdgePayload<H>> for EdgePayload<G>
where
    G: GraphGeometry,
    G::Edge: FromGeometry<H::Edge>,
    H: GraphGeometry,
{
    fn from_interior_geometry(edge: EdgePayload<H>) -> Self {
        EdgePayload {
            geometry: edge.geometry.into_geometry(),
            arc: edge.arc,
        }
    }
}

impl<G> Payload for EdgePayload<G>
where
    G: GraphGeometry,
{
    type Key = EdgeKey;
    type Attribute = G::Edge;
    type Storage = SlotStorage<Self>;
}

/// Face payload.
///
/// Contains the face attribute of `GraphGeometry`.
#[derivative(Clone, Copy, Debug, Hash)]
#[derive(Derivative)]
pub struct FacePayload<G>
where
    G: GraphGeometry,
{
    #[derivative(Debug = "ignore", Hash = "ignore")]
    pub geometry: G::Face,
    pub(in crate::graph) arc: ArcKey,
}

impl<G> FacePayload<G>
where
    G: GraphGeometry,
{
    pub(in crate::graph) fn new(arc: ArcKey, geometry: G::Face) -> Self {
        FacePayload { geometry, arc }
    }
}

impl<G, H> FromInteriorGeometry<FacePayload<H>> for FacePayload<G>
where
    G: GraphGeometry,
    G::Face: FromGeometry<H::Face>,
    H: GraphGeometry,
{
    fn from_interior_geometry(face: FacePayload<H>) -> Self {
        FacePayload {
            geometry: face.geometry.into_geometry(),
            arc: face.arc,
        }
    }
}

impl<G> Payload for FacePayload<G>
where
    G: GraphGeometry,
{
    type Key = FaceKey;
    type Attribute = G::Face;
    type Storage = SlotStorage<Self>;
}
