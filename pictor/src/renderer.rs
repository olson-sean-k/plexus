use wgpu::{
    Adapter, BackendBit, Device, DeviceDescriptor, Instance, PresentMode, Queue,
    RequestAdapterOptions, Surface, SwapChain, SwapChainDescriptor, TextureFormat, TextureUsage,
};
use winit::window::Window;

use crate::harness::{ConfigureStage, RenderStage};

pub struct Renderer {
    pub window: Window,
    _instance: Instance,
    _adapter: Adapter,
    pub surface: Surface,
    pub device: Device,
    queue: Queue,
    pub swap_chain_descriptor: SwapChainDescriptor,
    pub swap_chain: SwapChain,
}

impl Renderer {
    pub async fn try_from_window(window: Window) -> Result<Self, ()> {
        let instance = Instance::new(BackendBit::PRIMARY);
        let surface = unsafe { instance.create_surface(&window) };
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .ok_or_else(|| ())?;
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    shader_validation: true,
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|_| ())?;
        let dimensions = window.inner_size();
        let swap_chain_descriptor = SwapChainDescriptor {
            usage: TextureUsage::OUTPUT_ATTACHMENT,
            format: TextureFormat::Bgra8UnormSrgb,
            present_mode: PresentMode::Mailbox,
            width: dimensions.width,
            height: dimensions.height,
        };
        let swap_chain = device.create_swap_chain(&surface, &swap_chain_descriptor);
        Ok(Renderer {
            window,
            _instance: instance,
            _adapter: adapter,
            surface,
            device,
            queue,
            swap_chain_descriptor,
            swap_chain,
        })
    }
}

impl ConfigureStage for Renderer {
    fn device(&self) -> &Device {
        &self.device
    }

    fn queue(&self) -> &Queue {
        &self.queue
    }

    fn swap_chain_descriptor(&self) -> &SwapChainDescriptor {
        &self.swap_chain_descriptor
    }
}

impl RenderStage for Renderer {
    fn device(&self) -> &Device {
        &self.device
    }

    fn queue(&self) -> &Queue {
        &self.queue
    }
}
