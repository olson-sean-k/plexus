use nalgebra::{Isometry3, Matrix4, Orthographic3, Perspective3, Point3, Vector3};
use std::sync::LazyLock;
use wgpu::SurfaceConfiguration;

#[rustfmt::skip]
static OPENGL_TO_WGPU_TRANSFORM: LazyLock<Matrix4<f32>> = LazyLock::new(|| {
    Matrix4::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.0,
        0.0, 0.0, 0.5, 1.0,
    )
});

pub enum Projection {
    Perspective(Perspective3<f32>),
    Orthographic(Orthographic3<f32>),
}

impl Projection {
    pub fn perspective(aspect: f32, fov: f32, near: f32, far: f32) -> Self {
        Projection::Perspective(Perspective3::new(aspect, fov, near, far))
    }

    pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        Projection::Orthographic(Orthographic3::new(left, right, bottom, top, near, far))
    }
}

impl AsRef<Matrix4<f32>> for Projection {
    fn as_ref(&self) -> &Matrix4<f32> {
        match *self {
            Projection::Perspective(ref perspective) => perspective.as_matrix(),
            Projection::Orthographic(ref orthographic) => orthographic.as_matrix(),
        }
    }
}

pub struct Camera {
    pub projection: Projection,
    view: Isometry3<f32>,
}

impl Camera {
    pub fn look_at(&mut self, from: &Point3<f32>, to: &Point3<f32>) {
        self.view = Isometry3::look_at_rh(from, to, &Vector3::y());
    }

    pub fn reproject(&mut self, surface: &SurfaceConfiguration) {
        match self.projection {
            Projection::Perspective(ref mut perspective) => {
                perspective.set_aspect(surface.width as f32 / surface.height as f32);
            }
            Projection::Orthographic(ref mut orthographic) => {
                let inverse = surface.height as f32 / surface.width as f32;
                let radius = (orthographic.right() - orthographic.left()) * inverse * 0.5;
                orthographic.set_bottom_and_top(-radius, radius);
            }
        }
    }

    pub fn transform(&self) -> Matrix4<f32> {
        *OPENGL_TO_WGPU_TRANSFORM * self.projection.as_ref() * self.view.to_homogeneous()
    }
}

impl From<Projection> for Camera {
    fn from(projection: Projection) -> Self {
        Camera {
            projection,
            view: Isometry3::look_at_rh(
                &Point3::new(0.0, 0.0, 1.0),
                &Point3::origin(),
                &Vector3::y(),
            ),
        }
    }
}
