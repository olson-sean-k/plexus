// Macros require that crate roots are explicitly imported.
#![allow(clippy::single_component_path_imports)]

use plexus::integration::{nalgebra, theon};

use decorum::hash::FloatHash;
use gfx; // Imported for use in macro invocations.
use nalgebra::{Matrix4, Point3, Scalar, Vector3, Vector4};
use num::NumCast;
use std::hash::{Hash, Hasher};
use theon::adjunct::Map;

pub use self::pipeline::*;

trait IntoArray {
    type Output;

    fn into_array(self) -> Self::Output;
}

impl<T> IntoArray for Matrix4<T>
where
    T: Copy + Scalar,
{
    type Output = [[T; 4]; 4];

    #[rustfmt::skip]
    fn into_array(self) -> Self::Output {
        [
            [self[ 0], self[ 1], self[ 2], self[ 3]],
            [self[ 4], self[ 5], self[ 6], self[ 7]],
            [self[ 8], self[ 9], self[10], self[11]],
            [self[12], self[13], self[14], self[15]],
        ]
    }
}

impl<T> IntoArray for Point3<T>
where
    T: Scalar,
{
    type Output = [T; 3];

    fn into_array(self) -> Self::Output {
        self.coords.into()
    }
}

impl<T> IntoArray for Vector3<T>
where
    T: Scalar,
{
    type Output = [T; 3];

    fn into_array(self) -> Self::Output {
        self.into()
    }
}

impl<T> IntoArray for Vector4<T>
where
    T: Scalar,
{
    type Output = [T; 4];

    fn into_array(self) -> Self::Output {
        self.into()
    }
}

gfx_pipeline! {
    pipeline {
        buffer: gfx::VertexBuffer<Vertex> = (),
        transform: gfx::ConstantBuffer<Transform> = "transform",
        viewpoint: gfx::Global<[f32; 4]> = "u_viewpoint",
        camera: gfx::Global<[[f32; 4]; 4]> = "u_camera",
        model: gfx::Global<[[f32; 4]; 4]> = "u_model",
        color: gfx::RenderTarget<gfx::format::Rgba8> = "f_target0",
        depth: gfx::DepthTarget<gfx::format::Depth> = gfx::preset::depth::LESS_EQUAL_WRITE,
    }
}

gfx_constant_struct! {
    Transform {
        viewpoint: [f32; 4] = "u_viewpoint",
        camera: [[f32; 4]; 4] = "u_camera",
        model: [[f32; 4]; 4] = "u_model",
    }
}

impl Transform {
    pub fn new<T>(viewpoint: Point3<T>, camera: Matrix4<T>, model: Matrix4<T>) -> Self
    where
        T: NumCast + Scalar,
    {
        let viewpoint = viewpoint.map(crate::num_cast);
        Transform {
            viewpoint: [viewpoint[0], viewpoint[1], viewpoint[2], 1.0],
            camera: camera.map(crate::num_cast).into_array(),
            model: model.map(crate::num_cast).into_array(),
        }
    }
}

gfx_vertex_struct! {
    Vertex {
        position: [f32; 3] = "a_position",
        normal: [f32; 3] = "a_normal",
        color: [f32; 4] = "a_color",
    }
}

impl Vertex {
    pub fn new<T>(position: Point3<T>, normal: Vector3<T>, color: Vector4<T>) -> Self
    where
        T: NumCast + Scalar,
    {
        Vertex {
            position: position.map(crate::num_cast).into_array(),
            normal: normal.map(crate::num_cast).into_array(),
            color: color.map(crate::num_cast).into_array(),
        }
    }
}

impl Eq for Vertex {}

// TODO: The `gfx_vertex_struct` macro derives a `PartialEq` implementation
//       that may conflict with `Hash`.
#[allow(clippy::derive_hash_xor_eq)]
impl Hash for Vertex {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.position.hash(state);
        self.normal.hash(state);
        self.color.hash(state);
    }
}
