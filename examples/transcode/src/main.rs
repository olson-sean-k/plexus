use plexus::integration::nalgebra;
use plexus::integration::theon;

use nalgebra::Point3;
use pictor::pipeline::{self, Vertex};
use plexus::graph::MeshGraph;
use plexus::prelude::*;
use plexus::primitive::UnboundedPolygon;
use plexus::U3;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::io::BufReader;
use theon::space::{EuclideanSpace, FiniteDimensional, Vector, VectorSpace};
use tobj::Mesh;

type E3 = Point3<f32>;

trait MeshExt {
    fn into_raw_buffers<S>(self) -> (Vec<UnboundedPolygon<u32>>, Vec<S>)
    where
        S: EuclideanSpace + FiniteDimensional<N = U3>,
        Vector<S>: VectorSpace<Scalar = f32>;
}

impl MeshExt for Mesh {
    fn into_raw_buffers<S>(self) -> (Vec<UnboundedPolygon<u32>>, Vec<S>)
    where
        S: EuclideanSpace + FiniteDimensional<N = U3>,
        Vector<S>: VectorSpace<Scalar = f32>,
    {
        let Mesh {
            indices,
            num_face_indices: arities,
            positions: scalars,
            ..
        } = self;
        let mut polygons = Vec::new();
        let mut positions = Vec::new();
        let mut keys = HashMap::with_capacity(indices.len());
        let mut start = 0usize;
        for arity in arities {
            let arity = usize::try_from(arity).expect("overflow");
            let end = start + arity;
            let mut polygon = Vec::with_capacity(arity);
            for index in &indices[start..end] {
                let index = usize::try_from(*index).expect("overflow");
                let key = if let Some(key) = keys.get(&index) {
                    *key
                }
                else {
                    let key = positions.len();
                    positions.push(S::from_xyz(
                        *scalars.get(index * 3).expect("scalar"),
                        *scalars.get((index * 3) + 1).expect("scalar"),
                        *scalars.get((index * 3) + 2).expect("scalar"),
                    ));
                    keys.insert(index, key);
                    key
                };
                polygon.push(u32::try_from(key).expect("overflow"));
            }
            polygons.push(UnboundedPolygon::try_from_slice(&polygon).expect("polygon"));
            start = end;
        }
        (polygons, positions)
    }
}

fn main() {
    let from = Point3::new(-4.0, 6.0, 4.0);
    let to = Point3::new(0.0, 1.0, 0.0);
    pipeline::render_mesh_buffer_with(from, to, move || {
        // Read OBJ model data into graphs.
        let obj: &[u8] = include_bytes!("../../../data/teapot.obj");
        let (models, _) = tobj::load_obj_buf(&mut BufReader::new(obj), false, |_| {
            Ok((Vec::new(), HashMap::new()))
        })
        .expect("obj");
        let mut graphs: Vec<_> = models
            .into_iter()
            .map(|model| {
                let (indices, positions) = model.mesh.into_raw_buffers::<E3>();
                MeshGraph::<E3>::from_raw_buffers(indices, positions)
            })
            .collect::<Result<_, _>>()
            .expect("graphs");

        // TODO: This particular model seems to corrupt the resulting graph! The
        //       arc circulator for a particular vertex never terminates. For
        //       example, computing the vertex normal hangs. This does not seem
        //       to happen with other similar models.
        // Extract the graph read from the default model and convert it to a
        // buffer.
        let graph = graphs.get_mut(0).expect("teapot");
        graph.triangulate(); // TODO: This is very buggy!
        graph
            .to_mesh_by_face_with(|face, vertex| Vertex {
                position: vertex.position().into_homogeneous().into(),
                normal: face.normal().unwrap().into_homogeneous().into(),
                color: [1.0, 0.6, 0.2, 1.0],
            })
            .expect("buffer")
    });
}
