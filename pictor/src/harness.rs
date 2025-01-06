use futures::executor::{self, LocalPool, LocalSpawner};
use futures::task::LocalSpawn;
use std::cmp;
use std::fmt::Debug;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use crate::renderer::Renderer;

struct Executor {
    pool: LocalPool,
    spawner: LocalSpawner,
}

impl Executor {
    pub fn new() -> Self {
        let pool = LocalPool::new();
        let spawner = pool.spawner();
        Executor { pool, spawner }
    }

    pub fn flush(&mut self) {
        self.pool.run_until_stalled()
    }

    pub fn spawner(&self) -> &impl LocalSpawn {
        &self.spawner
    }
}

#[derive(Clone, Copy, Default)]
pub enum Reaction {
    #[default]
    Continue,
    Abort,
}

pub trait ConfigureStage {
    fn device(&self) -> &wgpu::Device;

    fn queue(&self) -> &wgpu::Queue;

    fn surface_configuration(&self) -> &wgpu::SurfaceConfiguration;
}

pub trait RenderStage {
    fn device(&self) -> &wgpu::Device;

    fn queue(&self) -> &wgpu::Queue;
}

pub trait Application: 'static + Sized {
    type Configuration: Sized;
    type Error: Debug;

    fn configure(
        configuration: Self::Configuration,
        stage: &impl ConfigureStage,
    ) -> Result<Self, Self::Error>;

    fn react(&mut self, event: WindowEvent) -> Reaction {
        let _ = event;
        Reaction::Continue
    }

    fn resize(&mut self, stage: &impl ConfigureStage);

    fn render(
        &mut self,
        stage: &impl RenderStage,
        view: &wgpu::TextureView,
        spawn: &impl LocalSpawn,
    );
}

pub fn run<T, F>(configuration: T::Configuration, f: F)
where
    T: Application,
    F: FnOnce(&EventLoop<()>) -> Window,
{
    struct Run<'window, T> {
        executor: Executor,
        application: T,
        window: &'window Window,
        renderer: Renderer<'window>,
    }

    impl<'window, T> ApplicationHandler<()> for Run<'window, T>
    where
        T: Application,
    {
        fn resumed(&mut self, _reactor: &ActiveEventLoop) {}

        fn window_event(&mut self, reactor: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
            match event {
                WindowEvent::RedrawRequested => {
                    let frame = self.renderer.surface.get_current_texture().unwrap();
                    let view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    self.application
                        .render(&self.renderer, &view, self.executor.spawner());
                    frame.present();
                }
                WindowEvent::CloseRequested => {
                    reactor.exit();
                }
                WindowEvent::Resized(dimensions) => {
                    self.renderer.surface_configuration.width = cmp::max(1, dimensions.width);
                    self.renderer.surface_configuration.height = cmp::max(1, dimensions.height);
                    self.application.resize(&self.renderer);
                    self.renderer
                        .surface
                        .configure(&self.renderer.device, &self.renderer.surface_configuration);
                    self.window.request_redraw();
                }
                _ => {
                    if let Reaction::Abort = self.application.react(event) {
                        reactor.exit();
                    }
                }
            }
            // TODO: This should probably be done in reaction to much more specific events.
            self.executor.flush();
        }
    }

    let executor = Executor::new();
    let reactor = EventLoop::new().unwrap();
    reactor.set_control_flow(ControlFlow::Poll);
    let window = f(&reactor);
    let renderer = executor::block_on(Renderer::try_from_window(&window)).unwrap();
    let application = T::configure(configuration, &renderer).unwrap();
    reactor
        .run_app(&mut Run {
            executor,
            application,
            window: &window,
            renderer,
        })
        .unwrap();
}
