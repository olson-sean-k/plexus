pub mod borrow;
pub mod dijkstra;
pub mod storage;
pub mod traverse;
pub mod view;

use thiserror::Error;

use crate::entity::storage::{Dispatch, Key, Storage};

#[derive(Debug, Eq, Error, PartialEq)]
pub enum EntityError {
    #[error("required entity not found")]
    EntityNotFound,
    #[error("data operation failed")]
    Data,
}

pub trait Entity: Sized {
    type Key: Key;
    type Storage: Default + Dispatch<Self> + Storage<Self>;
}

pub trait Payload: Entity {
    type Data;

    fn get(&self) -> &Self::Data;

    fn get_mut(&mut self) -> &mut Self::Data;
}
