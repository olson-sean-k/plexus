use plexus::integration::nalgebra;
use plexus::integration::theon;

use nalgebra::Point3;
use pictor::pipeline::{self, Vertex};
use plexus::prelude::*;
use plexus::primitive;
use plexus::primitive::generate::{Normal, Position};
use plexus::primitive::sphere::UvSphere;
use theon::space::{EuclideanSpace, VectorSpace};

type E3 = Point3<f32>;

fn main() {
    let from = Point3::new(0.0, 0.0, 3.0);
    let to = Point3::origin();
    pipeline::render_mesh_buffer_with(from, to, move || {
        let sphere = UvSphere::new(32, 16);
        primitive::zip_vertices((
            sphere.polygons::<Position<E3>>().triangulate(),
            sphere.polygons::<Normal<E3>>().triangulate(),
        ))
        .map_vertices(|(position, normal)| Vertex {
            position: position.into_homogeneous().into(),
            normal: normal.into_inner().into_homogeneous().into(),
            color: [1.0, 0.6, 0.2, 1.0],
        })
        .collect()
    });
}
