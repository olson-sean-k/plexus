pub mod borrow;
pub mod dijkstra;
pub mod storage;
pub mod traverse;
pub mod view;

use thiserror::Error;

use crate::entity::storage::{Dispatch, Key, Storage, Unjournaled};

#[derive(Debug, Error, PartialEq)]
pub enum EntityError {
    #[error("required entity not found")]
    EntityNotFound,
    #[error("data operation failed")]
    Data,
}

#[cfg(not(all(nightly, feature = "unstable")))]
pub trait Lifetime: 'static {}

#[cfg(not(all(nightly, feature = "unstable")))]
impl<T> Lifetime for T where T: 'static {}

#[cfg(all(nightly, feature = "unstable"))]
pub trait Lifetime {}

#[cfg(all(nightly, feature = "unstable"))]
impl<T> Lifetime for T {}

pub trait Entity: Copy + Lifetime + Sized {
    type Key: Key;
    type Storage: Default + Dispatch<Self> + Storage<Self> + Unjournaled;
}

pub trait Payload: Entity {
    type Data;

    fn get(&self) -> &Self::Data;

    fn get_mut(&mut self) -> &mut Self::Data;
}
