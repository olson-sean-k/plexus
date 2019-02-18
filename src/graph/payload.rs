use crate::geometry::convert::{FromGeometry, FromInteriorGeometry, IntoGeometry};
use crate::geometry::{Attribute, Geometry};
use crate::graph::storage::{ArcKey, EdgeKey, FaceKey, OpaqueKey, VertexKey};

pub trait Payload {
    type Key: OpaqueKey;
    type Attribute: Attribute;
}

#[derivative(Debug, Hash)]
#[derive(Clone, Derivative)]
pub struct VertexPayload<G>
where
    G: Geometry,
{
    #[derivative(Debug = "ignore", Hash = "ignore")]
    pub geometry: G::Vertex,
    pub(in crate::graph) arc: Option<ArcKey>,
}

impl<G> VertexPayload<G>
where
    G: Geometry,
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
    G: Geometry,
    G::Vertex: FromGeometry<H::Vertex>,
    H: Geometry,
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
    G: Geometry,
{
    type Key = VertexKey;
    type Attribute = G::Vertex;
}

// Unlike other topological structures, the vertex connectivity of `Arc`s is
// immutable and encoded within the key for each arc. A arc key
// consists of its source and destination vertex keys. This provides fast and
// reliable arc lookups, even when a mesh is in an inconsistent state.
// However, it also complicates basic mutations of vertices and arcs,
// requiring rekeying of `Arc`s.
//
// For this reason, `Arc` has no fields for storing its destination vertex key
// or opposite arc key, as these would be redundant.
#[derivative(Debug, Hash)]
#[derive(Clone, Derivative)]
pub struct ArcPayload<G>
where
    G: Geometry,
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
    G: Geometry,
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
    G: Geometry,
    G::Arc: FromGeometry<H::Arc>,
    H: Geometry,
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
    G: Geometry,
{
    type Key = ArcKey;
    type Attribute = G::Arc;
}

#[derivative(Debug, Hash)]
#[derive(Clone, Derivative)]
pub struct EdgePayload<G>
where
    G: Geometry,
{
    #[derivative(Debug = "ignore", Hash = "ignore")]
    pub geometry: G::Edge,
    pub(in crate::graph) arc: ArcKey,
}

impl<G> EdgePayload<G>
where
    G: Geometry,
{
    pub(in crate::graph) fn new(arc: ArcKey, geometry: G::Edge) -> Self {
        EdgePayload { geometry, arc }
    }
}

impl<G, H> FromInteriorGeometry<EdgePayload<H>> for EdgePayload<G>
where
    G: Geometry,
    G::Edge: FromGeometry<H::Edge>,
    H: Geometry,
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
    G: Geometry,
{
    type Key = EdgeKey;
    type Attribute = G::Edge;
}

#[derivative(Debug, Hash)]
#[derive(Clone, Derivative)]
pub struct FacePayload<G>
where
    G: Geometry,
{
    #[derivative(Debug = "ignore", Hash = "ignore")]
    pub geometry: G::Face,
    pub(in crate::graph) arc: ArcKey,
}

impl<G> FacePayload<G>
where
    G: Geometry,
{
    pub(in crate::graph) fn new(arc: ArcKey, geometry: G::Face) -> Self {
        FacePayload { geometry, arc }
    }
}

impl<G, H> FromInteriorGeometry<FacePayload<H>> for FacePayload<G>
where
    G: Geometry,
    G::Face: FromGeometry<H::Face>,
    H: Geometry,
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
    G: Geometry,
{
    type Key = FaceKey;
    type Attribute = G::Face;
}
