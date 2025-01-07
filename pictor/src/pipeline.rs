use bytemuck::{self, Pod, Zeroable};
use decorum::cmp::CanonicalEq;
use decorum::hash::CanonicalHash;
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
use wgpu::include_spirv_raw;
use wgpu::util::DeviceExt as _;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::keyboard::{Key, NamedKey};
use winit::window::Window;

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

#[derive(Clone, Copy, Debug)]
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
        self.position.hash_canonical(state);
        self.normal.hash_canonical(state);
        self.color.hash_canonical(state);
    }
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.position.eq_canonical(&other.position)
            && self.normal.eq_canonical(&other.normal)
            && self.color.eq_canonical(&other.color)
    }
}

// SAFETY: This type is inhabited, is `repr(C)`, has no padding, has no illegal bit patterns,
//         contains no pointers, and has only fields that are also `Pod`.
unsafe impl Pod for Vertex {}

// SAFETY: This type has a zeroed inhabitant.
unsafe impl Zeroable for Vertex {}

#[derive(Debug)]
struct RenderConfiguration {
    camera: Camera,
    from: Point3<f32>,
    buffer: MeshBuffer<Flat3<u32>, Vertex>,
}

#[derive(Debug)]
struct RenderApplication {
    camera: Camera,
    _viewpoint: wgpu::Buffer,
    transform: wgpu::Buffer,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    depth: wgpu::TextureView,
    n: u32,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
}

impl RenderApplication {
    fn configure_depth_buffer(stage: &impl ConfigureStage) -> wgpu::TextureView {
        stage
            .device()
            .create_texture(&wgpu::TextureDescriptor {
                label: None,
                view_formats: stage.surface_configuration().view_formats.as_slice(),
                size: wgpu::Extent3d {
                    width: stage.surface_configuration().width,
                    height: stage.surface_configuration().height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT,
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
        camera.reproject(stage.surface_configuration());
        let vertices = stage
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(buffer.as_vertex_slice()),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let indices = stage
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(buffer.as_index_slice()),
                usage: wgpu::BufferUsages::INDEX,
            });
        let transform = stage
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(camera.transform().as_slice()),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            });
        let viewpoint = stage
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(from.coords.as_slice()),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            });
        let depth = Self::configure_depth_buffer(stage);
        let bind_group_layout =
            stage
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(64),
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(12),
                            },
                            count: None,
                        },
                    ],
                });
        let bind_group = stage
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: transform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: viewpoint.as_entire_binding(),
                    },
                ],
            });
        let pipeline = stage
            .device()
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&stage.device().create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    },
                )),
                vertex: wgpu::VertexState {
                    module: unsafe {
                        &stage
                            .device()
                            .create_shader_module_spirv(&include_spirv_raw!("shader.spv.vert"))
                    },
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            #[allow(clippy::erasing_op)]
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 0 * 4 * 4,
                                shader_location: 0,
                            },
                            #[allow(clippy::identity_op)]
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 1 * 4 * 4,
                                shader_location: 1,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 2 * 4 * 4,
                                shader_location: 2,
                            },
                        ],
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: unsafe {
                        &stage
                            .device()
                            .create_shader_module_spirv(&include_spirv_raw!("shader.spv.frag"))
                    },
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: stage.surface_configuration().format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent::REPLACE,
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                multisample: Default::default(),
                multiview: None,
                cache: None,
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
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => Abort,
            _ => Continue,
        }
    }

    fn resize(&mut self, stage: &impl ConfigureStage) {
        self.camera.reproject(stage.surface_configuration());
        stage.queue().write_buffer(
            &self.transform,
            0,
            bytemuck::cast_slice(self.camera.transform().as_slice()),
        );
        self.depth = Self::configure_depth_buffer(stage);
    }

    fn render(&mut self, stage: &impl RenderStage, view: &wgpu::TextureView, _: &impl LocalSpawn) {
        let mut encoder = stage
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.2,
                        b: 0.3,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.draw_indexed(0..self.n, 0, 0..1);

        drop(pass); // Release `encoder`.
        stage.queue().submit(Some(encoder.finish()));
    }
}

pub fn render_mesh_buffer_with<F>(from: Point3<f32>, to: Point3<f32>, mut f: F)
where
    F: FnMut() -> MeshBuffer<Flat3<u32>, Vertex>,
{
    let camera = {
        let mut camera = Camera::from(Projection::perspective(1.0, FRAC_PI_4, 0.1, 8.0));
        //let mut camera = Camera::from(Projection::orthographic(-4.0, 4.0, -4.0, 4.0, -8.0, 8.0));
        camera.look_at(&from, &to);
        camera
    };
    harness::run::<RenderApplication, _>(move || {
        (
            Window::default_attributes().with_title("Plexus"),
            RenderConfiguration {
                camera: camera.clone(),
                from,
                buffer: f(),
            },
        )
    })
}
