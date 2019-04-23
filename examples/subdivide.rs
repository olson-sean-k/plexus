extern crate arrayvec;
extern crate decorum;
extern crate nalgebra;
extern crate plexus;

use decorum::R64;
use nalgebra::Point2;
use plexus::geometry::alias::VertexPosition;
use plexus::geometry::convert::AsPosition;
use plexus::geometry::Geometry;
use plexus::graph::{EdgeMidpoint, FaceView, MeshGraph};
use plexus::prelude::*;
use plexus::primitive::Triangle;
use smallvec::SmallVec;

pub fn circumscribe<G>(face: FaceView<&mut MeshGraph<G>, G>) -> FaceView<&mut MeshGraph<G>, G>
where
    G: EdgeMidpoint<Midpoint = VertexPosition<G>> + Geometry,
    G::Vertex: AsPosition,
{
    // Split each edge, stashing the vertex key and moving to the next arc.
    let arity = face.arity();
    let mut arc = face.into_arc();
    let mut splits = SmallVec::<[_; 4]>::with_capacity(arity);
    for _ in 0..arity {
        let vertex = arc.split_at_midpoint();
        splits.push(vertex.key());
        arc = vertex.into_outgoing_arc().into_next_arc();
    }
    // Split faces along the vertices from each arc split.
    let mut face = arc.into_face().unwrap();
    for (a, b) in splits.into_iter().perimeter() {
        face = face.split(ByKey(a), ByKey(b)).unwrap().into_face().unwrap();
    }
    // Return the central face of the subdivision.
    face
}

fn main() {
    // Create a graph from a right triangle.
    let mut graph = MeshGraph::<Point2<R64>>::from_raw_buffers(
        vec![Triangle::new(0usize, 1, 2)],
        vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
    )
    .unwrap();
    // Get the key of the singular face.
    let key = graph.faces().nth(0).unwrap().key();
    // Subdivide the face.
    let _ = circumscribe(graph.face_mut(key).unwrap());
}
