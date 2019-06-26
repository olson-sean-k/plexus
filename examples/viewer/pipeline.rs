use decorum;
use gfx;
use nalgebra::{Matrix4, Point3, Scalar, Vector3, Vector4};
use num::One;
use plexus::geometry::UnitGeometry;
use std::hash::{Hash, Hasher};

pub use self::pipeline::*;

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
    pub fn new(viewpoint: &Point3<f32>, camera: &Matrix4<f32>, model: &Matrix4<f32>) -> Self {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Transform {
            viewpoint: [viewpoint[0], viewpoint[1], viewpoint[2], 1.0],
            camera: [
                [camera[ 0], camera[ 1], camera[ 2], camera[ 3]],
                [camera[ 4], camera[ 5], camera[ 6], camera[ 7]],
                [camera[ 8], camera[ 9], camera[10], camera[11]],
                [camera[12], camera[13], camera[14], camera[15]],
            ],
            model: [
                [model[ 0], model[ 1], model[ 2], model[ 3]],
                [model[ 4], model[ 5], model[ 6], model[ 7]],
                [model[ 8], model[ 9], model[10], model[11]],
                [model[12], model[13], model[14], model[15]],
            ],
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
    pub fn new(position: &Point3<f32>, normal: &Vector3<f32>, color: &Vector4<f32>) -> Self {
        Vertex {
            position: [position[0], position[1], position[2]],
            normal: [normal[0], normal[1], normal[2]],
            color: [color[0], color[1], color[2], color[3]],
        }
    }
}

impl Hash for Vertex {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        decorum::hash_float_array(&self.position, state);
        decorum::hash_float_array(&self.normal, state);
        decorum::hash_float_array(&self.color, state);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Color4<T>(pub Vector4<T>)
where
    T: Scalar;

impl<T> Color4<T>
where
    T: Scalar,
{
    pub fn white() -> Self
    where
        T: One,
    {
        Color4(Vector4::repeat(One::one()))
    }
}

impl<T> AsRef<Vector4<T>> for Color4<T>
where
    T: Scalar,
{
    fn as_ref(&self) -> &Vector4<T> {
        &self.0
    }
}

impl<T> Default for Color4<T>
where
    T: One + Scalar,
{
    fn default() -> Self {
        Color4::white()
    }
}

impl<T> From<Vector4<T>> for Color4<T>
where
    T: Scalar,
{
    fn from(vector: Vector4<T>) -> Self {
        Color4(vector)
    }
}

impl<T> Into<Vector4<T>> for Color4<T>
where
    T: Scalar,
{
    fn into(self) -> Vector4<T> {
        self.0
    }
}

impl<T> UnitGeometry for Color4<T> where T: One + Scalar {}
