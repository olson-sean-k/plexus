use futures::executor::{self, LocalPool, LocalSpawner};
use futures::task::LocalSpawn;
use std::cmp;
use std::fmt::Debug;
use wgpu::{Device, Queue, SwapChainDescriptor, SwapChainTexture};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Window;

use crate::renderer::Renderer;

use ControlFlow::Exit;
use ControlFlow::Poll;

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

#[derive(Clone, Copy)]
pub enum Reaction {
    Continue,
    Abort,
}

impl Default for Reaction {
    fn default() -> Self {
        Reaction::Continue
    }
}

pub trait ConfigureStage {
    fn device(&self) -> &Device;

    fn queue(&self) -> &Queue;

    fn swap_chain_descriptor(&self) -> &SwapChainDescriptor;
}

pub trait RenderStage {
    fn device(&self) -> &Device;

    fn queue(&self) -> &Queue;
}

pub trait Application: 'static + Sized {
    type State: Sized;
    type Error: Debug;

    fn configure(state: Self::State, stage: &impl ConfigureStage) -> Result<Self, Self::Error>;

    fn react(&mut self, event: WindowEvent) -> Reaction {
        let _ = event;
        Reaction::Continue
    }

    fn resize(&mut self, stage: &impl ConfigureStage);

    fn render(
        &mut self,
        stage: &impl RenderStage,
        frame: &SwapChainTexture,
        spawn: &impl LocalSpawn,
    );
}

pub fn run<T, F>(state: T::State, f: F)
where
    T: Application,
    F: FnOnce(&EventLoop<()>) -> Window,
{
    let mut executor = Executor::new();
    let reactor = EventLoop::new();
    let mut renderer = executor::block_on(Renderer::try_from_window(f(&reactor))).unwrap();
    let mut application = T::configure(state, &renderer).unwrap();
    reactor.run(move |event, _, reaction| {
        *reaction = Poll;
        match event {
            Event::MainEventsCleared => {
                executor.flush();
            }
            Event::RedrawRequested(_) => {
                let swap_chain = &mut renderer.swap_chain;
                let frame = match swap_chain.get_current_frame() {
                    Ok(frame) => frame,
                    Err(_) => {
                        *swap_chain = renderer
                            .device
                            .create_swap_chain(&renderer.surface, &renderer.swap_chain_descriptor);
                        swap_chain.get_current_frame().unwrap()
                    }
                };
                application.render(&renderer, &frame.output, executor.spawner());
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    *reaction = Exit;
                }
                WindowEvent::Resized(dimensions) => {
                    renderer.swap_chain_descriptor.width = cmp::max(1, dimensions.width);
                    renderer.swap_chain_descriptor.height = cmp::max(1, dimensions.height);
                    application.resize(&renderer);
                    renderer.swap_chain = renderer
                        .device
                        .create_swap_chain(&renderer.surface, &renderer.swap_chain_descriptor);
                }
                _ => {
                    if let Reaction::Abort = application.react(event) {
                        *reaction = Exit;
                    }
                }
            },
            _ => {}
        }
    });
}
