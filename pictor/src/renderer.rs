use gfx::format::{Depth, Rgba8};
use gfx::handle::{DepthStencilView, RenderTargetView};
use gfx::state::{CullFace, FrontFace, RasterMethod, Rasterizer};
use gfx::traits::FactoryExt;
use gfx::{CommandBuffer, Device, Encoder, Factory, PipelineState, Primitive, Resources};
use gfx_device_gl;
use gfx_window_glutin;
use glutin::{NotCurrent, PossiblyCurrent, WindowedContext};
use plexus::buffer::MeshBuffer3;

use crate::pipeline::{self, Data, Meta, Transform, Vertex};

const CLEAR_COLOR: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

pub trait SwapBuffers {
    fn swap_buffers(&mut self) -> Result<(), ()>;
}

impl SwapBuffers for WindowedContext<PossiblyCurrent> {
    fn swap_buffers(&mut self) -> Result<(), ()> {
        match WindowedContext::swap_buffers(self) {
            Ok(_) => Ok(()),
            Err(_) => Err(()),
        }
    }
}

pub trait UpdateFrameBufferView<R>
where
    R: Resources,
{
    fn update_frame_buffer_view(
        &self,
        color: &mut RenderTargetView<R, Rgba8>,
        depth: &mut DepthStencilView<R, Depth>,
    );
}

impl UpdateFrameBufferView<gfx_device_gl::Resources> for WindowedContext<PossiblyCurrent> {
    fn update_frame_buffer_view(
        &self,
        color: &mut RenderTargetView<gfx_device_gl::Resources, Rgba8>,
        depth: &mut DepthStencilView<gfx_device_gl::Resources, Depth>,
    ) {
        gfx_window_glutin::update_views(self, color, depth);
    }
}

pub trait Binding {
    type Window: SwapBuffers + UpdateFrameBufferView<Self::Resources>;
    type Resources: Resources;
    type Factory: Factory<Self::Resources>;
    type CommandBuffer: CommandBuffer<Self::Resources>;
    type Device: Device<Resources = Self::Resources, CommandBuffer = Self::CommandBuffer>;
}

pub enum GlutinBinding {}

impl Binding for GlutinBinding {
    type Window = WindowedContext<PossiblyCurrent>;
    type Resources = gfx_device_gl::Resources;
    type Factory = gfx_device_gl::Factory;
    type CommandBuffer = gfx_device_gl::CommandBuffer;
    type Device = gfx_device_gl::Device;
}

pub struct Renderer<R>
where
    R: Binding,
{
    pub window: R::Window,
    pub factory: R::Factory,
    device: R::Device,
    encoder: Encoder<R::Resources, R::CommandBuffer>,
    state: PipelineState<R::Resources, Meta>,
    data: Data<R::Resources>,
}

impl Renderer<GlutinBinding> {
    pub fn from_glutin_window(window: WindowedContext<NotCurrent>) -> Self {
        let (window, device, mut factory, color, depth) = gfx_window_glutin::init_existing(window);
        let encoder = factory.create_command_buffer().into();
        Renderer::new(window, factory, device, encoder, color, depth)
    }
}

impl<R> Renderer<R>
where
    R: Binding,
{
    fn new(
        window: R::Window,
        mut factory: R::Factory,
        device: R::Device,
        encoder: Encoder<R::Resources, R::CommandBuffer>,
        color: RenderTargetView<R::Resources, Rgba8>,
        depth: DepthStencilView<R::Resources, Depth>,
    ) -> Self {
        let shaders = factory
            .create_shader_set(
                include_bytes!("shader.v.glsl"),
                include_bytes!("shader.f.glsl"),
            )
            .unwrap();
        let state = factory
            .create_pipeline_state(
                &shaders,
                Primitive::TriangleList,
                Rasterizer {
                    method: RasterMethod::Fill,
                    front_face: FrontFace::CounterClockwise,
                    cull_face: CullFace::Back,
                    offset: None,
                    samples: None,
                },
                pipeline::new(),
            )
            .unwrap();
        let data = Data {
            buffer: factory.create_vertex_buffer(&[]),
            transform: factory.create_constant_buffer(1),
            viewpoint: [0.0; 4],
            camera: [[0.0; 4]; 4],
            model: [[0.0; 4]; 4],
            color,
            depth,
        };
        Renderer {
            window,
            factory,
            device,
            encoder,
            state,
            data,
        }
    }

    pub fn set_transform(&mut self, transform: &Transform) -> Result<(), ()> {
        self.data.camera = transform.camera;
        self.data.model = transform.model;
        match self
            .encoder
            .update_buffer(&self.data.transform, &[*transform], 0)
        {
            Ok(_) => Ok(()),
            Err(_) => Err(()),
        }
    }

    pub fn draw_mesh_buffer(&mut self, buffer: &MeshBuffer3<u32, Vertex>) {
        let (buffer, slice) = self
            .factory
            .create_vertex_buffer_with_slice(buffer.as_vertex_slice(), buffer.as_index_slice());
        self.data.buffer = buffer;
        self.encoder.draw(&slice, &self.state, &self.data);
    }

    pub fn clear(&mut self) {
        self.encoder.clear(&self.data.color, CLEAR_COLOR);
        self.encoder.clear_depth(&self.data.depth, 1.0);
    }

    pub fn flush(&mut self) -> Result<(), ()> {
        self.encoder.flush(&mut self.device);
        match self.window.swap_buffers().and_then(|_| {
            self.device.cleanup();
            Ok(())
        }) {
            Ok(_) => Ok(()),
            Err(_) => Err(()),
        }
    }

    pub fn update_frame_buffer_view(&mut self) {
        self.window
            .update_frame_buffer_view(&mut self.data.color, &mut self.data.depth);
    }
}
