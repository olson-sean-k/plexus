#![allow(clippy::iter_nth_zero)]

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use nalgebra::Point3;
use plexus::geometry::AsPositionMut;
use plexus::graph::{EdgeMidpoint, FaceView, GraphData, MeshGraph};
use plexus::prelude::*;
use plexus::primitive::Tetragon;
use smallvec::SmallVec;

const DEPTH: usize = 8;

type E3 = Point3<f64>;

pub trait Ambo<G> {
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

fn tetragon() -> MeshGraph<E3> {
    // Create a graph from a tetragon.
    MeshGraph::from(Tetragon::from([
        (1.0, 0.0, -1.0),
        (-1.0, 0.0, -1.0),
        (-1.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
    ]))
}

fn subdivide(mut graph: MeshGraph<E3>) {
    // Get the face of the tetragon.
    let key = graph.faces().nth(0).unwrap().key();
    let mut face = graph.face_mut(key).unwrap();

    // Subdivide and extrude the face repeatedly.
    for _ in 0..DEPTH {
        face = face.ambo().extrude_with_offset(0.5).unwrap();
    }
}

#[allow(unused)]
fn benchmark(criterion: &mut Criterion) {
    criterion.bench_function("subdivide", move |bencher| {
        bencher.iter_batched(tetragon, subdivide, BatchSize::SmallInput)
    });
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
