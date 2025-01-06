#![allow(clippy::iter_nth_zero)]

use nalgebra::Point3;
use pictor::pipeline::{self, Vertex};
use plexus::geometry::AsPositionMut;
use plexus::graph::{ClosedView, EdgeMidpoint, FaceView, GraphData, MeshGraph};
use plexus::prelude::*;
use plexus::primitive::Tetragon;
use smallvec::SmallVec;
use theon::space::{EuclideanSpace, VectorSpace};

type E3 = Point3<f32>;

pub trait Ambo<G>: ClosedView {
    #[must_use]
    fn ambo(self) -> Self;
}

impl<'a, G> Ambo<G> for FaceView<&'a mut MeshGraph<G>>
where
    G: EdgeMidpoint + GraphData,
    G::Vertex: AsPositionMut,
{
    // Subdivide the face with a polygon formed from vertices at the midpoints
    // of the edges of the face.
    fn ambo(self) -> Self {
        // Split each edge, stashing the vertex key and moving to the next arc.
        let arity = self.arity();
        let mut arc = self.into_arc();
        let mut splits = SmallVec::<[_; 4]>::with_capacity(arity);
        for _ in 0..arity {
            let vertex = arc.split_at_midpoint();
            splits.push(vertex.key());
            arc = vertex.into_outgoing_arc().into_next_arc();
        }
        // Split faces along the vertices from each arc split.
        let mut face = arc.into_face().unwrap();
        for (a, b) in splits.into_iter().perimeter() {
            face = face.split(a, b).unwrap().into_face().unwrap();
        }
        // Return the face forming the similar polygon.
        face
    }
}

fn main() {
    let from = Point3::new(-0.9, 3.1, 2.4);
    let to = Point3::new(0.0, 1.0, 0.0);
    pipeline::render_mesh_buffer_with(from, to, || {
        // Create a graph from a tetragon.
        let mut graph = MeshGraph::<E3>::from(Tetragon::from([
            (1.0, 0.0, -1.0),
            (-1.0, 0.0, -1.0),
            (-1.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
        ]));
        // Get the face of the tetragon.
        let key = graph.faces().nth(0).unwrap().key();
        let mut face = graph.face_mut(key).unwrap();

        // Subdivide and extrude the face repeatedly.
        for _ in 0..5 {
            face = face.ambo().extrude_with_offset(0.5).unwrap();
        }

        // Convert the graph into a buffer.
        graph.triangulate();
        graph
            .to_mesh_by_face_with(|face, vertex| Vertex {
                position: vertex.position().into_homogeneous().into(),
                normal: face.normal().unwrap().into_homogeneous().into(),
                color: [1.0, 0.6, 0.2, 1.0],
            })
            .unwrap()
    });
}
