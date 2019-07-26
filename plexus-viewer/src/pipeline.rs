use decorum;
use gfx;
use nalgebra::{Matrix4, Point3, Scalar, Vector3, Vector4};
use num::{NumCast, ToPrimitive};
use std::hash::{Hash, Hasher};
use theon::ops::Map;

pub use self::pipeline::*;

trait IntoArray {
    type Output;

    fn into_array(self) -> Self::Output;
}

impl<T> IntoArray for Matrix4<T>
where
    T: Scalar,
{
    type Output = [[T; 4]; 4];

    #[cfg_attr(rustfmt, rustfmt_skip)]
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
        [self[0], self[1], self[2]]
    }
}

impl<T> IntoArray for Vector3<T>
where
    T: Scalar,
{
    type Output = [T; 3];

    fn into_array(self) -> Self::Output {
        [self[0], self[1], self[2]]
    }
}

impl<T> IntoArray for Vector4<T>
where
    T: Scalar,
{
    type Output = [T; 4];

    fn into_array(self) -> Self::Output {
        [self[0], self[1], self[2], self[3]]
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
    pub fn new<T>(viewpoint: &Point3<T>, camera: &Matrix4<T>, model: &Matrix4<T>) -> Self
    where
        T: Scalar + ToPrimitive,
    {
        let viewpoint = viewpoint.map(num_cast);
        Transform {
            viewpoint: [viewpoint[0], viewpoint[1], viewpoint[2], 1.0],
            camera: camera.map(num_cast).into_array(),
            model: model.map(num_cast).into_array(),
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
    pub fn new<T>(position: &Point3<T>, normal: &Vector3<T>, color: &Vector4<T>) -> Self
    where
        T: Scalar + ToPrimitive,
    {
        Vertex {
            position: position.map(num_cast).into_array(),
            normal: normal.map(num_cast).into_array(),
            color: color.map(num_cast).into_array(),
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

fn num_cast<T, U>(value: T) -> U
where
    T: ToPrimitive,
    U: NumCast,
{
    <U as NumCast>::from(value).expect("numeric cast")
}
