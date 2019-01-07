use crate::geometry::convert::{FromGeometry, FromInteriorGeometry, IntoGeometry};
use crate::geometry::{Attribute, Geometry};
use crate::graph::storage::{EdgeKey, FaceKey, OpaqueKey, VertexKey};

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
    pub(in crate::graph) edge: Option<EdgeKey>,
}

impl<G> Vertex<G>
where
    G: Geometry,
{
    pub(in crate::graph) fn new(geometry: G::Vertex) -> Self {
        Vertex {
            geometry,
            edge: None,
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
            edge: vertex.edge,
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

// Unlike other topological structures, the vertex connectivity of `Edge`s is
// immutable and encoded within the key for each half-edge. A half-edge key
// consists of its source and destination vertex keys. This provides fast and
// reliable half-edge lookups, even when a mesh is in an inconsistent state.
// However, it also complicates basic mutations of vertices and half-edges,
// requiring rekeying of `Edge`s.
//
// For this reason, `Edge` has no fields for storing its destination vertex key
// or opposite edge key, as these would be redundant.
#[derivative(Debug, Hash)]
#[derive(Clone, Derivative)]
pub struct Edge<G>
where
    G: Geometry,
{
    #[derivative(Debug = "ignore", Hash = "ignore")]
    pub geometry: G::Edge,
    pub(in crate::graph) next: Option<EdgeKey>,
    pub(in crate::graph) previous: Option<EdgeKey>,
    pub(in crate::graph) face: Option<FaceKey>,
}

impl<G> Edge<G>
where
    G: Geometry,
{
    pub(in crate::graph) fn new(geometry: G::Edge) -> Self {
        Edge {
            geometry,
            next: None,
            previous: None,
            face: None,
        }
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
            next: edge.next,
            previous: edge.previous,
            face: edge.face,
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
    pub(in crate::graph) edge: EdgeKey,
}

impl<G> Face<G>
where
    G: Geometry,
{
    pub(in crate::graph) fn new(edge: EdgeKey, geometry: G::Face) -> Self {
        Face { geometry, edge }
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
            edge: face.edge,
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
