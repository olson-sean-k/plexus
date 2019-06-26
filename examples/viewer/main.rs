#![allow(unknown_lints)] // Allow clippy lints.

#[macro_use]
extern crate gfx;

mod camera;
mod pipeline;
mod renderer;

use glutin::{
    ContextBuilder, ControlFlow, Event, EventsLoop, GlWindow, WindowBuilder, WindowEvent,
};
use nalgebra::{Matrix4, Point3};
use plexus::encoding::ply::PointEncoding;
use plexus::graph::MeshGraph;
use plexus::prelude::*;
use std::f32::consts::PI;

use crate::camera::Camera;
use crate::pipeline::{Transform, Vertex};
use crate::renderer::Renderer;

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 576;

fn main() {
    // Establish an event loop and renderer for the application.
    let mut event_loop = EventsLoop::new();
    let mut renderer = Renderer::from_glutin_window(
        GlWindow::new(
            WindowBuilder::new()
                .with_title("Plexus Viewer")
                .with_dimensions((WIDTH, HEIGHT).into())
                .with_resizable(false),
            ContextBuilder::new(),
            &event_loop,
        )
        .expect("window"),
    );

    // Compute constant transformations used in the vertex shader.
    let transform = {
        let viewpoint = Point3::new(0.0, -8.0, 4.0);
        let camera = {
            let mut camera = Camera::new(WIDTH as f32 / HEIGHT as f32, PI / 4.0, 0.1, 100.0);
            camera.look_at(&viewpoint, &Point3::new(0.0, 0.0, 1.0));
            camera
        };
        Transform::new(&viewpoint, &camera.transform(), &Matrix4::identity())
    };

    // Read a PLY file into a graph. This graph can be further manipulated as
    // needed. Convert the graph into a buffer that can be rendered.
    let graph = {
        let ply: &[u8] = include_bytes!("../../data/teapot.ply");
        MeshGraph::<Point3<f32>>::from_ply(PointEncoding::<Point3<f32>>::default(), ply)
            .expect("teapot")
            .0
    };
    let buffer = graph
        .to_mesh_buffer_by_face_with(|face, vertex| Vertex::new(&vertex.geometry, &face.normal()))
        .expect("buffer");

    // Start the application and exit when the window is closed.
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
