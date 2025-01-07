use winit::window::Window;

use crate::harness::{ConfigureStage, RenderStage};

pub struct Renderer<'window> {
    //pub window: Window,
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    pub surface: wgpu::Surface<'window>,
    pub surface_configuration: wgpu::SurfaceConfiguration,
    pub device: wgpu::Device,
    queue: wgpu::Queue,
}

impl<'window> Renderer<'window> {
    pub async fn try_from_window(window: &'window Window) -> Result<Self, ()> {
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
        let dimensions = window.inner_size();
        let surface_configuration = surface
            .get_default_config(&adapter, dimensions.width, dimensions.height)
            .unwrap();
        //let swap_chain_descriptor = wgpu::SwapChainDescriptor {
        //    usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        //    format: wgpu::TextureFormat::Bgra8UnormSrgb,
        //    present_mode: wgpu::PresentMode::Mailbox,
        //    width: dimensions.width,
        //    height: dimensions.height,
        //};
        //let swap_chain = device.create_swap_chain(&surface, &swap_chain_descriptor);
        Ok(Renderer {
            //window,
            _instance: instance,
            _adapter: adapter,
            surface,
            surface_configuration,
            device,
            queue,
        })
    }
}

impl<'window> ConfigureStage for Renderer<'window> {
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

impl<'window> RenderStage for Renderer<'window> {
    fn device(&self) -> &wgpu::Device {
        &self.device
    }

    fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}
