pub mod borrow;
pub mod storage;
pub mod traverse;
pub mod view;

use crate::network::storage::{AsStorage, Get, OpaqueKey, Remove, Sequence};

pub trait Entity: Copy + Sized {
    type Key: OpaqueKey;
    type Storage: Default + Get<Self> + Remove<Self> + Sequence<Self>;
}

pub trait Fuse<M, T>
where
    M: AsStorage<T>,
    T: Entity,
{
    type Output: AsStorage<T>;

    fn fuse(self, source: M) -> Self::Output;
}
