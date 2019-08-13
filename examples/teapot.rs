use decorum::R64;
use nalgebra::Point3;
use pictor::pipeline::Vertex;
use pictor::{self, Color4};
use plexus::encoding::ply::{FromPly, PositionEncoding};
use plexus::graph::MeshGraph;

type E3 = Point3<R64>;

fn main() {
    let from = Point3::new(0.0, -10.0, 4.0);
    let to = Point3::new(0.0, 0.0, 1.0);
    pictor::draw_with(from, to, move || {
        // Read PLY data into a graph.
        let ply: &[u8] = include_bytes!("../data/teapot.ply");
        let encoding = PositionEncoding::<E3>::default();
        let (graph, _) = MeshGraph::<E3>::from_ply(encoding, ply).expect("teapot");

        // Convert the graph into a buffer that can be rendered for viewing.
        graph
            .to_mesh_buffer_by_vertex_with(|vertex| {
                Vertex::new(
                    *vertex.position(),
                    vertex.normal().unwrap(),
                    Color4::white().into(),
                )
            })
            .expect("buffer")
    });
}
