extern crate arrayvec;
extern crate decorum;
extern crate nalgebra;
extern crate plexus;

use decorum::R64;
use nalgebra::Point2;
use plexus::geometry::convert::AsPosition;
use plexus::geometry::Geometry;
use plexus::graph::geometry::alias::VertexPosition;
use plexus::graph::geometry::EdgeMidpoint;
use plexus::graph::{EdgeView, FaceView, GraphError, MeshGraph};
use plexus::prelude::*;

pub fn subdivide_triangular_face<G>(face: FaceView<&mut MeshGraph<G>, G>) -> Result<(), GraphError>
where
    G: EdgeMidpoint<Midpoint = VertexPosition<G>> + Geometry,
    G::Vertex: AsPosition,
{
    // Bail if the face is not triangular.
    let arity = face.arity();
    if arity != 3 {
        return Err(GraphError::ArityConflict {
            expected: 3,
            actual: arity,
        });
    }
    // Split each edge, stashing the vertex key and moving to the next edge.
    let split = |edge: EdgeView<_, G>| -> Result<_, GraphError> {
        let vertex = edge.split()?;
        let x = vertex.key();
        Ok((x, vertex.into_outgoing_edge().into_next_edge()))
    };
    let (a, edge) = split(face.into_edge())?;
    let (b, edge) = split(edge)?;
    let (c, edge) = split(edge)?;
    // Bisect along the vertices from each edge split.
    edge.into_face()
        .unwrap()
        .bisect(a, b)?
        .into_face()
        .unwrap()
        .bisect(b, c)?
        .into_face()
        .unwrap()
        .bisect(c, a)
        .map(|_| ())
}

fn main() {
    // Create a graph from a right triangle.
    let mut graph = MeshGraph::<Point2<R64>>::from_raw_buffers_with_arity(
        vec![0u32, 1, 2],
        vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        3,
    )
    .unwrap();
    // Get the key of the singular face.
    let abc = graph.faces().nth(0).unwrap().key();
    // Subdivide the face.
    subdivide_triangular_face(graph.face_mut(abc).unwrap()).unwrap();
}
