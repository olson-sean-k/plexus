use decorum;
use gfx;
use nalgebra::{Matrix4, Point3, Vector3};
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
    }
}

impl Vertex {
    pub fn new(position: &Point3<f32>, normal: &Vector3<f32>) -> Self {
        Vertex {
            position: [position[0], position[1], position[2]],
            normal: [normal[0], normal[1], normal[2]],
        }
    }
}

impl Default for Vertex {
    fn default() -> Self {
        Vertex {
            position: Default::default(),
            normal: [1.0, 0.0, 0.0],
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
    }
}
