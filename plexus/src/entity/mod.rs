pub mod borrow;
pub mod dijkstra;
pub mod storage;
pub mod traverse;
pub mod view;

use thiserror::Error;

use crate::entity::storage::{Get, OpaqueKey, Remove, Sequence};

#[derive(Debug, Error, PartialEq)]
pub enum EntityError {
    #[error("required entity not found")]
    EntityNotFound,
    #[error("geometric operation failed")]
    Geometry,
}

pub trait Entity: Copy + Sized {
    type Key: OpaqueKey;
    type Storage: Default + Get<Self> + Remove<Self> + Sequence<Self>;
}
