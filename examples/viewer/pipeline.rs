#![allow(derive_hash_xor_eq)]

use decorum;
use gfx;
use nalgebra::{Matrix4, Point3, Scalar, Vector4};
use num::One;
use rand;
use rand::distributions::{Distribution, Standard};
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};

pub use self::pipeline::*;

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Color4<T>(Vector4<T>)
where
    T: Scalar;

impl<T> Color4<T>
where
    T: Scalar,
{
    pub fn new(r: T, g: T, b: T, a: T) -> Self {
        Color4(Vector4::new(r, g, b, a))
    }
}

impl<T> Color4<T>
where
    T: One + Scalar,
{
    pub fn white() -> Self {
        Color4::new(T::one(), T::one(), T::one(), T::one())
    }
}

impl<T> Color4<T>
where
    T: One + Scalar,
{
    pub fn random() -> Self
    where
        Standard: Distribution<T>,
    {
        Color4::new(
            rand::random::<T>(),
            rand::random::<T>(),
            rand::random::<T>(),
            T::one(),
        )
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

impl<T> Deref for Color4<T>
where
    T: Scalar,
{
    type Target = Vector4<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Color4<T>
where
    T: Scalar,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

gfx_pipeline! {
    pipeline {
        buffer: gfx::VertexBuffer<Vertex> = (),
        transform: gfx::ConstantBuffer<Transform> = "transform",
        camera: gfx::Global<[[f32; 4]; 4]> = "u_camera",
        model: gfx::Global<[[f32; 4]; 4]> = "u_model",
        color: gfx::RenderTarget<gfx::format::Rgba8> = "f_target0",
        depth: gfx::DepthTarget<gfx::format::DepthStencil> = gfx::preset::depth::LESS_EQUAL_WRITE,
    }
}

gfx_constant_struct! {
    Transform {
        camera: [[f32; 4]; 4] = "u_camera",
        model: [[f32; 4]; 4] = "u_model",
    }
}

impl Transform {
    pub fn new(camera: &Matrix4<f32>, model: &Matrix4<f32>) -> Self {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Transform {
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
        color: [f32; 4] = "a_color",
    }
}

impl Vertex {
    pub fn new(position: &Point3<f32>, color: &Color4<f32>) -> Self {
        Vertex {
            position: [position[0], position[1], position[2]],
            color: [color[0], color[1], color[2], color[3]],
        }
    }
}

impl Default for Vertex {
    fn default() -> Self {
        Vertex {
            position: Default::default(),
            color: Default::default(),
        }
    }
}

impl Eq for Vertex {}

impl Hash for Vertex {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        decorum::hash_float_array(&self.position, state);
        decorum::hash_float_array(&self.color, state);
    }
}
