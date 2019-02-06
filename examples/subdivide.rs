extern crate arrayvec;
extern crate decorum;
extern crate nalgebra;
extern crate plexus;

use decorum::R64;
use nalgebra::Point2;
use plexus::geometry::alias::VertexPosition;
use plexus::geometry::compose::EdgeMidpoint;
use plexus::geometry::convert::AsPosition;
use plexus::geometry::Geometry;
use plexus::graph::{FaceView, GraphError, MeshGraph};
use plexus::prelude::*;
use smallvec::SmallVec;

pub fn subdivide<G>(
    face: FaceView<&mut MeshGraph<G>, G>,
) -> Result<FaceView<&mut MeshGraph<G>, G>, GraphError>
where
    G: EdgeMidpoint<Midpoint = VertexPosition<G>> + Geometry,
    G::Vertex: AsPosition,
{
    // Split each edge, stashing the vertex key and moving to the next
    // arc-edge.
    let arity = face.arity();
    let mut arc = face.into_arc();
    let mut splits = SmallVec::<[_; 4]>::with_capacity(arity);
    for _ in 0..arity {
        let vertex = arc.split_at_midpoint()?;
        splits.push(vertex.key());
        arc = vertex.into_outgoing_arc().into_next_arc();
    }
    // Bisect along the vertices from each edge split.
    let mut face = arc.into_face().unwrap();
    for (a, b) in splits.into_iter().perimeter() {
        face = face.bisect(ByKey(a), ByKey(b))?.into_face().unwrap();
    }
    // Return the central face of the subdivision.
    Ok(face)
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
    subdivide(graph.face_mut(abc).unwrap()).unwrap();
}
