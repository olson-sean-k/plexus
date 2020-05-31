pub mod borrow;
pub mod storage;
pub mod traverse;
pub mod view;

use crate::entity::storage::{Get, OpaqueKey, Remove, Sequence};

pub trait Entity: Copy + Sized {
    type Key: OpaqueKey;
    type Storage: Default + Get<Self> + Remove<Self> + Sequence<Self>;
}
