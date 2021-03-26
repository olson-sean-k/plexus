use plexus::integration::nalgebra;
use plexus::integration::theon;

use bytemuck::{self, Pod, Zeroable};
use decorum::cmp::FloatEq;
use decorum::hash::FloatHash;
use fool::and;
use futures::task::LocalSpawn;
use nalgebra::{Point3, Scalar, Vector3, Vector4};
use num::{self, One};
use plexus::buffer::MeshBuffer;
use plexus::geometry::UnitGeometry;
use plexus::index::Flat3;
use rand::distributions::{Distribution, Standard};
use rand::{self, Rng};
use std::f32::consts::FRAC_PI_4;
use std::hash::{Hash, Hasher};
use std::mem;
use theon::adjunct::Extend;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    include_spirv, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BlendDescriptor, Buffer, BufferAddress, BufferSize,
    BufferUsage, Color, ColorStateDescriptor, ColorWrite, CommandEncoderDescriptor,
    CompareFunction, CullMode, DepthStencilStateDescriptor, Extent3d, FrontFace, IndexFormat,
    InputStepMode, LoadOp, Operations, PipelineLayoutDescriptor, PrimitiveTopology,
    ProgrammableStageDescriptor, RasterizationStateDescriptor, RenderPassColorAttachmentDescriptor,
    RenderPassDepthStencilAttachmentDescriptor, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, ShaderStage, SwapChainTexture, TextureDescriptor, TextureDimension,
    TextureFormat, TextureUsage, TextureView, VertexAttributeDescriptor, VertexBufferDescriptor,
    VertexFormat, VertexStateDescriptor,
};
use winit::event::{ElementState, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::window::WindowBuilder;

use crate::camera::{Camera, Projection};
use crate::harness::{self, Application, ConfigureStage, Reaction, RenderStage};

use Reaction::Abort;
use Reaction::Continue;

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
        T: One,
        Standard: Distribution<T>,
        Vector3<T>: Extend<Vector4<T>, Item = T>,
    {
        let mut rng = rand::thread_rng();
        Color4(Vector3::from_fn(|_, _| rng.gen()).extend(One::one()))
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

impl<T> From<Color4<T>> for Vector4<T>
where
    T: Scalar,
{
    fn from(color: Color4<T>) -> Self {
        color.0
    }
}

impl<T> UnitGeometry for Color4<T> where T: One + Scalar {}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Vertex {
    pub position: [f32; 4],
    pub normal: [f32; 4],
    pub color: [f32; 4],
}

impl Eq for Vertex {}

impl Hash for Vertex {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.position.float_hash(state);
        self.normal.float_hash(state);
        self.color.float_hash(state);
    }
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        and!(
            self.position.float_eq(&other.position),
            self.normal.float_eq(&other.normal),
            self.color.float_eq(&other.color),
        )
    }
}

unsafe impl Pod for Vertex {}

unsafe impl Zeroable for Vertex {}

struct RenderConfiguration {
    camera: Camera,
    from: Point3<f32>,
    buffer: MeshBuffer<Flat3<u32>, Vertex>,
}

struct RenderApplication {
    camera: Camera,
    _viewpoint: Buffer,
    transform: Buffer,
    vertices: Buffer,
    indices: Buffer,
    depth: TextureView,
    n: u32,
    bind_group: BindGroup,
    pipeline: RenderPipeline,
}

impl RenderApplication {
    fn configure_depth_buffer(stage: &impl ConfigureStage) -> TextureView {
        stage
            .device()
            .create_texture(&TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: stage.swap_chain_descriptor().width,
                    height: stage.swap_chain_descriptor().height,
                    depth: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Depth32Float,
                usage: TextureUsage::COPY_DST | TextureUsage::OUTPUT_ATTACHMENT,
            })
            .create_view(&Default::default())
    }
}

impl Application for RenderApplication {
    type Configuration = RenderConfiguration;
    type Error = ();

    fn configure(
        configuration: Self::Configuration,
        stage: &impl ConfigureStage,
    ) -> Result<Self, Self::Error> {
        let RenderConfiguration {
            mut camera,
            from,
            buffer,
        } = configuration;
        camera.reproject(stage.swap_chain_descriptor());
        let vertices = stage.device().create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(buffer.as_vertex_slice()),
            usage: BufferUsage::VERTEX,
        });
        let indices = stage.device().create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(buffer.as_index_slice()),
            usage: BufferUsage::INDEX,
        });
        let transform = stage.device().create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(camera.transform().as_slice()),
            usage: BufferUsage::COPY_DST | BufferUsage::UNIFORM,
        });
        let viewpoint = stage.device().create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(from.coords.as_slice()),
            usage: BufferUsage::COPY_DST | BufferUsage::UNIFORM,
        });
        let depth = Self::configure_depth_buffer(stage);
        let bind_group_layout =
            stage
                .device()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStage::VERTEX,
                            ty: BindingType::UniformBuffer {
                                dynamic: false,
                                min_binding_size: BufferSize::new(64),
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStage::VERTEX,
                            ty: BindingType::UniformBuffer {
                                dynamic: false,
                                min_binding_size: BufferSize::new(12),
                            },
                            count: None,
                        },
                    ],
                });
        let bind_group = stage.device().create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: transform.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: viewpoint.as_entire_binding(),
                },
            ],
        });
        let pipeline =
            stage
                .device()
                .create_render_pipeline(&RenderPipelineDescriptor {
                    label: None,
                    layout: Some(&stage.device().create_pipeline_layout(
                        &PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&bind_group_layout],
                            push_constant_ranges: &[],
                        },
                    )),
                    vertex_stage: ProgrammableStageDescriptor {
                        module: &stage
                            .device()
                            .create_shader_module(include_spirv!("shader.spv.vert")),
                        entry_point: "main",
                    },
                    fragment_stage: Some(ProgrammableStageDescriptor {
                        module: &stage
                            .device()
                            .create_shader_module(include_spirv!("shader.spv.frag")),
                        entry_point: "main",
                    }),
                    vertex_state: VertexStateDescriptor {
                        index_format: IndexFormat::Uint32,
                        vertex_buffers: &[VertexBufferDescriptor {
                            stride: mem::size_of::<Vertex>() as BufferAddress,
                            step_mode: InputStepMode::Vertex,
                            attributes: &[
                                #[allow(clippy::erasing_op)]
                                VertexAttributeDescriptor {
                                    format: VertexFormat::Float4,
                                    offset: 0 * 4 * 4,
                                    shader_location: 0,
                                },
                                #[allow(clippy::identity_op)]
                                VertexAttributeDescriptor {
                                    format: VertexFormat::Float4,
                                    offset: 1 * 4 * 4,
                                    shader_location: 1,
                                },
                                VertexAttributeDescriptor {
                                    format: VertexFormat::Float4,
                                    offset: 2 * 4 * 4,
                                    shader_location: 2,
                                },
                            ],
                        }],
                    },
                    primitive_topology: PrimitiveTopology::TriangleList,
                    rasterization_state: Some(RasterizationStateDescriptor {
                        front_face: FrontFace::Ccw,
                        cull_mode: CullMode::Back,
                        ..Default::default()
                    }),
                    color_states: &[ColorStateDescriptor {
                        format: stage.swap_chain_descriptor().format,
                        color_blend: BlendDescriptor::REPLACE,
                        alpha_blend: BlendDescriptor::REPLACE,
                        write_mask: ColorWrite::ALL,
                    }],
                    depth_stencil_state: Some(DepthStencilStateDescriptor {
                        format: TextureFormat::Depth32Float,
                        depth_write_enabled: true,
                        depth_compare: CompareFunction::Less,
                        stencil: Default::default(),
                    }),
                    alpha_to_coverage_enabled: false,
                    sample_count: 1,
                    sample_mask: !0,
                });
        Ok(RenderApplication {
            camera,
            _viewpoint: viewpoint,
            transform,
            vertices,
            indices,
            depth,
            n: buffer.as_index_slice().len() as u32,
            bind_group,
            pipeline,
        })
    }

    fn react(&mut self, event: WindowEvent) -> Reaction {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        virtual_keycode: Some(VirtualKeyCode::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => Abort,
            _ => Continue,
        }
    }

    fn resize(&mut self, stage: &impl ConfigureStage) {
        self.camera.reproject(stage.swap_chain_descriptor());
        stage.queue().write_buffer(
            &self.transform,
            0,
            bytemuck::cast_slice(self.camera.transform().as_slice()),
        );
        self.depth = Self::configure_depth_buffer(stage);
    }

    fn render(&mut self, stage: &impl RenderStage, frame: &SwapChainTexture, _: &impl LocalSpawn) {
        let mut encoder = stage
            .device()
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            color_attachments: &[RenderPassColorAttachmentDescriptor {
                attachment: &frame.view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color {
                        r: 0.1,
                        g: 0.2,
                        b: 0.3,
                        a: 1.0,
                    }),
                    store: true,
                },
            }],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachmentDescriptor {
                attachment: &self.depth,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: true,
                }),
                stencil_ops: None,
            }),
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_index_buffer(self.indices.slice(..));
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.draw_indexed(0..self.n, 0, 0..1);
        drop(pass); // Release `encoder`.
        stage.queue().submit(Some(encoder.finish()));
    }
}

pub fn render_mesh_buffer_with<F>(from: Point3<f32>, to: Point3<f32>, f: F)
where
    F: FnOnce() -> MeshBuffer<Flat3<u32>, Vertex>,
{
    let camera = {
        let mut camera = Camera::from(Projection::perspective(1.0, FRAC_PI_4, 0.1, 8.0));
        //let mut camera = Camera::from(Projection::orthographic(-4.0, 4.0, -4.0, 4.0, -8.0, 8.0));
        camera.look_at(&from, &to);
        camera
    };
    let buffer = f();
    harness::run::<RenderApplication, _>(
        RenderConfiguration {
            camera,
            from,
            buffer,
        },
        |reactor| {
            WindowBuilder::default()
                .with_title("Plexus")
                .build(reactor)
                .unwrap()
        },
    );
}
