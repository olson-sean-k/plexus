use plexus::integration::nalgebra;
use plexus::integration::theon;

use nalgebra::Point3;
use pictor::pipeline::{self, Vertex};
use plexus::encoding::ply::{FromPly, PositionEncoding};
use plexus::graph::MeshGraph;
use theon::space::{EuclideanSpace, VectorSpace};

type E3 = Point3<f32>;

fn main() {
    let from = Point3::new(0.0, -6.0, 4.0);
    let to = Point3::new(0.0, 0.0, 1.0);
    pipeline::render_mesh_buffer_with(from, to, move || {
        // Read PLY data into a graph.
        let ply: &[u8] = include_bytes!("../../../data/teapot.ply");
        let encoding = PositionEncoding::<E3>::default();
        let (graph, _) = MeshGraph::<E3>::from_ply(encoding, ply).expect("teapot");

        // Convert the graph into a buffer.
        graph
            .to_mesh_by_vertex_with(|vertex| Vertex {
                position: vertex.position().into_homogeneous().into(),
                normal: vertex.normal().unwrap().into_homogeneous().into(),
                color: [1.0, 0.6, 0.2, 1.0],
            })
            .expect("buffer")
    });
}
