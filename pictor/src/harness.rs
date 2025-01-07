use futures::executor::{self, LocalPool, LocalSpawner};
use futures::task::LocalSpawn;
use std::cmp;
use std::fmt::Debug;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::renderer::Renderer;

#[derive(Debug)]
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

#[derive(Debug)]
struct Activity<T> {
    window: Arc<Window>,
    renderer: Renderer<'static>,
    application: T,
}

impl<T> Activity<T>
where
    T: Application,
{
    pub fn bind_and_configure(window: Window, configuration: T::Configuration) -> Self {
        let window = Arc::new(window);
        let renderer = executor::block_on(Renderer::try_from_window(window.clone())).unwrap();
        let application = T::configure(configuration, &renderer).unwrap();
        Activity {
            window,
            renderer,
            application,
        }
    }
}

#[derive(Debug)]
struct Harness<T, F> {
    executor: Executor,
    f: F,
    activity: Option<Activity<T>>,
}

impl<T, F> Harness<T, F>
where
    T: Application,
{
    fn redraw(&mut self) {
        if let Some(activity) = self.activity.as_mut() {
            let frame = activity.renderer.surface.get_current_texture().unwrap();
            let view = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            activity
                .application
                .render(&activity.renderer, &view, self.executor.spawner());
            frame.present();
        }
    }

    fn resize(&mut self, dimensions: PhysicalSize<u32>) {
        if let Some(activity) = self.activity.as_mut() {
            activity.renderer.surface_configuration.width = cmp::max(1, dimensions.width);
            activity.renderer.surface_configuration.height = cmp::max(1, dimensions.height);
            activity.application.resize(&activity.renderer);
            activity.renderer.surface.configure(
                &activity.renderer.device,
                &activity.renderer.surface_configuration,
            );
            activity.window.request_redraw();
        }
    }
}

impl<T, F> ApplicationHandler<()> for Harness<T, F>
where
    T: Application,
    F: FnMut() -> (WindowAttributes, T::Configuration),
{
    fn resumed(&mut self, reactor: &ActiveEventLoop) {
        let (window, configuration) = (self.f)();
        self.activity.replace(Activity::bind_and_configure(
            reactor.create_window(window).unwrap(),
            configuration,
        ));
    }

    fn suspended(&mut self, _reactor: &ActiveEventLoop) {
        self.activity.take();
    }

    fn window_event(&mut self, reactor: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::RedrawRequested => {
                self.redraw();
            }
            WindowEvent::CloseRequested => {
                reactor.exit();
            }
            WindowEvent::Resized(dimensions) => {
                self.resize(dimensions);
            }
            _ => {
                if let Some(activity) = self.activity.as_mut() {
                    if let Reaction::Abort = activity.application.react(event) {
                        reactor.exit();
                    }
                }
            }
        }
        self.executor.flush();
    }
}

#[derive(Clone, Copy, Debug, Default)]
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

pub fn run<T, F>(f: F)
where
    T: Application,
    F: FnMut() -> (WindowAttributes, T::Configuration),
{
    let executor = Executor::new();
    let reactor = EventLoop::new().unwrap();
    reactor.set_control_flow(ControlFlow::Poll);
    reactor
        .run_app(&mut Harness::<T, _> {
            executor,
            f,
            activity: None,
        })
        .unwrap();
}
