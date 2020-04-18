#[macro_use]
extern crate gfx;

pub mod camera;
pub mod pipeline;
pub mod renderer;

use plexus::integration::{nalgebra, theon};

use glutin::{ContextBuilder, ControlFlow, Event, EventsLoop, WindowBuilder, WindowEvent};
use nalgebra::{Matrix4, Point3, Scalar, Vector4};
use num::cast;
use num::{NumCast, One};
use plexus::buffer::MeshBuffer3;
use plexus::UnitGeometry;
use rand::distributions::{Distribution, Standard};
use rand::{self, Rng};
use std::f32::consts::PI;
use theon::adjunct::Map;

use crate::camera::Camera;
use crate::pipeline::{Transform, Vertex};
use crate::renderer::Renderer;

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

    pub fn random() -> Self
    where
        Standard: Distribution<T>,
    {
        let mut rng = rand::thread_rng();
        Color4(Vector4::from_fn(|_, _| rng.gen()))
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

pub fn draw_with<T, F>(from: Point3<T>, to: Point3<T>, f: F)
where
    T: NumCast + Scalar,
    F: FnOnce() -> MeshBuffer3<u32, Vertex>,
{
    const WIDTH: u32 = 640;
    const HEIGHT: u32 = 640;
    const ASPECT: f32 = WIDTH as f32 / HEIGHT as f32;

    let from = from.map(num_cast);
    let to = to.map(num_cast);
    let buffer = f();
    let transform = Transform::new(
        from,
        Camera::new(ASPECT, PI / 4.0, 0.1, 100.0)
            .look_at(from, to)
            .transform(),
        Matrix4::identity(),
    );

    let mut event_loop = EventsLoop::new();
    let mut renderer = Renderer::from_glutin_window(
        ContextBuilder::new()
            .build_windowed(
                WindowBuilder::new()
                    .with_title("Plexus")
                    .with_dimensions((WIDTH, HEIGHT).into())
                    .with_resizable(false),
                &event_loop,
            )
            .expect("window"),
    );
    event_loop.run_forever(move |event| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => ControlFlow::Break,
        _ => {
            renderer.clear();
            renderer.set_transform(&transform).unwrap();
            renderer.draw_mesh_buffer(&buffer);
            renderer.flush().unwrap();
            ControlFlow::Continue
        }
    });
}

fn num_cast<T, U>(value: T) -> U
where
    T: NumCast,
    U: NumCast,
{
    cast::cast(value).expect("numeric cast failed")
}
