use std::borrow::Borrow;
use winit::window::Window;

use crate::harness::{ConfigureStage, RenderStage};

#[derive(Debug)]
pub struct Renderer<'window> {
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    pub surface: wgpu::Surface<'window>,
    pub surface_configuration: wgpu::SurfaceConfiguration,
    pub device: wgpu::Device,
    queue: wgpu::Queue,
}

impl<'window> Renderer<'window> {
    pub async fn try_from_window<T>(window: T) -> Result<Self, ()>
    where
        T: Borrow<Window> + Into<wgpu::SurfaceTarget<'window>>,
    {
        let dimensions = window.borrow().inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .ok_or(())?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::SPIRV_SHADER_PASSTHROUGH,
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|_| ())?;
        let surface_configuration = surface
            .get_default_config(&adapter, dimensions.width, dimensions.height)
            .unwrap();
        Ok(Renderer {
            _instance: instance,
            _adapter: adapter,
            surface,
            surface_configuration,
            device,
            queue,
        })
    }
}

impl ConfigureStage for Renderer<'_> {
    fn device(&self) -> &wgpu::Device {
        &self.device
    }

    fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    fn surface_configuration(&self) -> &wgpu::SurfaceConfiguration {
        &self.surface_configuration
    }
}

impl RenderStage for Renderer<'_> {
    fn device(&self) -> &wgpu::Device {
        &self.device
    }

    fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}
