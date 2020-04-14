#![allow(clippy::iter_nth_zero)]

use plexus::integration::nalgebra;

use decorum::N64;
use nalgebra::Point2;
use plexus::graph::{EdgeMidpoint, FaceView, GraphGeometry, MeshGraph};
use plexus::prelude::*;
use plexus::primitive::NGon;
use plexus::AsPosition;
use smallvec::SmallVec;

pub fn circumscribe<G>(face: FaceView<&mut MeshGraph<G>>) -> FaceView<&mut MeshGraph<G>>
where
    G: EdgeMidpoint + GraphGeometry,
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
    let mut graph = MeshGraph::<Point2<N64>>::from_raw_buffers(
        vec![NGon([0usize, 1, 2])],
        vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
    )
    .unwrap();
    // Get the key of the singular face.
    let key = graph.faces().nth(0).unwrap().key();
    // Subdivide the face.
    let _ = circumscribe(graph.face_mut(key).unwrap());
}
