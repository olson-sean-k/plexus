#![allow(unknown_lints)] // Allow clippy lints.

extern crate decorum;
#[macro_use]
extern crate gfx;
extern crate gfx_device_gl;
extern crate gfx_window_glutin;
extern crate glutin;
extern crate nalgebra;
extern crate num;
extern crate plexus;
extern crate rand;

mod camera;
mod pipeline;
mod renderer;

use glutin::{
    ContextBuilder, ControlFlow, Event, EventsLoop, GlWindow, WindowBuilder, WindowEvent,
};
use nalgebra::{Matrix4, Point3, Scalar};
use plexus::buffer::MeshBuffer;
use plexus::geometry::{Attribute, Geometry};
use plexus::graph::Mesh;
use plexus::prelude::*;
use plexus::primitive::sphere::UvSphere;

use camera::Camera;
use pipeline::{Color4, Transform, Vertex};
use renderer::{GlutinRenderer, Renderer};

impl<T> Attribute for Color4<T> where T: Scalar {}

struct FaceColorGeometry;

impl Geometry for FaceColorGeometry {
    type Vertex = Point3<f32>;
    type Edge = ();
    type Face = Color4<f32>;
}

fn new_mesh_buffer() -> MeshBuffer<u32, Vertex> {
    let mut mesh = UvSphere::new(32, 32)
        .polygons_with_position()
        .triangulate()
        .collect::<Mesh<FaceColorGeometry>>();
    for mut face in mesh.orphan_faces() {
        face.geometry = Color4::random();
    }
    mesh.to_mesh_buffer_by_face_with(|face, vertex| Vertex::new(&vertex.geometry, &face.geometry))
        .unwrap()
}

fn new_camera(aspect: f32) -> Camera<f32> {
    let mut camera = Camera::new(aspect, 45.0, -1.0, 2.0);
    camera.look_at(&Point3::new(2.0, 1.0, 2.0), &Point3::origin());
    camera
}

fn new_renderer(width: u32, height: u32) -> (EventsLoop, Renderer<GlutinRenderer>) {
    let seat = EventsLoop::new();
    let window = GlWindow::new(
        WindowBuilder::new()
            .with_title("Plexus Viewer")
            .with_dimensions(width, height),
        ContextBuilder::new(),
        &seat,
    ).unwrap();
    (seat, Renderer::from_glutin_window(window))
}

fn main() {
    let (mut seat, mut renderer) = new_renderer(1024, 576);
    let mut camera = new_camera(1024.0 / 576.0);
    let mesh = new_mesh_buffer();
    seat.run_forever(|event| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::Closed,
                ..
            } => {
                return ControlFlow::Break;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(width, height),
                ..
            } => {
                camera = new_camera(width as f32 / height as f32);
                renderer.update_frame_buffer_view();
            }
            _ => {}
        }
        renderer.clear();
        renderer
            .set_transform(&Transform::new(&camera.transform(), &Matrix4::identity()))
            .unwrap();
        renderer.draw_mesh_buffer(&mesh);
        renderer.flush().unwrap();
        ControlFlow::Continue
    });
}
