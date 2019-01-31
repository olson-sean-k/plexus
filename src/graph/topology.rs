use crate::geometry::convert::{FromGeometry, FromInteriorGeometry, IntoGeometry};
use crate::geometry::{Attribute, Geometry};
use crate::graph::storage::{ArcKey, EdgeKey, FaceKey, OpaqueKey, VertexKey};

pub trait Topological {
    type Key: OpaqueKey;
    type Attribute: Attribute;
}

#[derivative(Debug, Hash)]
#[derive(Clone, Derivative)]
pub struct Vertex<G>
where
    G: Geometry,
{
    #[derivative(Debug = "ignore", Hash = "ignore")]
    pub geometry: G::Vertex,
    pub(in crate::graph) arc: Option<ArcKey>,
}

impl<G> Vertex<G>
where
    G: Geometry,
{
    pub(in crate::graph) fn new(geometry: G::Vertex) -> Self {
        Vertex {
            geometry,
            arc: None,
        }
    }
}

impl<G, H> FromInteriorGeometry<Vertex<H>> for Vertex<G>
where
    G: Geometry,
    G::Vertex: FromGeometry<H::Vertex>,
    H: Geometry,
{
    fn from_interior_geometry(vertex: Vertex<H>) -> Self {
        Vertex {
            geometry: vertex.geometry.into_geometry(),
            arc: vertex.arc,
        }
    }
}

impl<G> Topological for Vertex<G>
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
pub struct Arc<G>
where
    G: Geometry,
{
    #[derivative(Debug = "ignore", Hash = "ignore")]
    pub geometry: G::Arc,
    pub(in crate::graph) next: Option<ArcKey>,
    pub(in crate::graph) previous: Option<ArcKey>,
    pub(in crate::graph) face: Option<FaceKey>,
}

impl<G> Arc<G>
where
    G: Geometry,
{
    pub(in crate::graph) fn new(geometry: G::Arc) -> Self {
        Arc {
            geometry,
            next: None,
            previous: None,
            face: None,
        }
    }
}

impl<G, H> FromInteriorGeometry<Arc<H>> for Arc<G>
where
    G: Geometry,
    G::Arc: FromGeometry<H::Arc>,
    H: Geometry,
{
    fn from_interior_geometry(arc: Arc<H>) -> Self {
        Arc {
            geometry: arc.geometry.into_geometry(),
            next: arc.next,
            previous: arc.previous,
            face: arc.face,
        }
    }
}

impl<G> Topological for Arc<G>
where
    G: Geometry,
{
    type Key = ArcKey;
    type Attribute = G::Arc;
}

#[derivative(Debug, Hash)]
#[derive(Clone, Derivative)]
pub struct Edge<G>
where
    G: Geometry,
{
    #[derivative(Debug = "ignore", Hash = "ignore")]
    pub geometry: G::Edge,
    pub(in crate::graph) arc: ArcKey,
}

impl<G> Edge<G>
where
    G: Geometry,
{
    pub(in crate::graph) fn new(arc: ArcKey, geometry: G::Edge) -> Self {
        Edge { geometry, arc }
    }
}

impl<G, H> FromInteriorGeometry<Edge<H>> for Edge<G>
where
    G: Geometry,
    G::Edge: FromGeometry<H::Edge>,
    H: Geometry,
{
    fn from_interior_geometry(edge: Edge<H>) -> Self {
        Edge {
            geometry: edge.geometry.into_geometry(),
            arc: edge.arc,
        }
    }
}

impl<G> Topological for Edge<G>
where
    G: Geometry,
{
    type Key = EdgeKey;
    type Attribute = G::Edge;
}

#[derivative(Debug, Hash)]
#[derive(Clone, Derivative)]
pub struct Face<G>
where
    G: Geometry,
{
    #[derivative(Debug = "ignore", Hash = "ignore")]
    pub geometry: G::Face,
    pub(in crate::graph) arc: ArcKey,
}

impl<G> Face<G>
where
    G: Geometry,
{
    pub(in crate::graph) fn new(arc: ArcKey, geometry: G::Face) -> Self {
        Face { geometry, arc }
    }
}

impl<G, H> FromInteriorGeometry<Face<H>> for Face<G>
where
    G: Geometry,
    G::Face: FromGeometry<H::Face>,
    H: Geometry,
{
    fn from_interior_geometry(face: Face<H>) -> Self {
        Face {
            geometry: face.geometry.into_geometry(),
            arc: face.arc,
        }
    }
}

impl<G> Topological for Face<G>
where
    G: Geometry,
{
    type Key = FaceKey;
    type Attribute = G::Face;
}
