mod core;

pub use self::core::{Bind, Core};

pub trait ConsistencyContract {}

pub struct Consistent;

impl ConsistencyContract for Consistent {}

pub struct Indeterminate;

impl ConsistencyContract for Indeterminate {}

pub trait Container {
    type Contract: ConsistencyContract;
}

impl<'a, T> Container for &'a T
where
    T: Container,
{
    type Contract = <T as Container>::Contract;
}

impl<'a, T> Container for &'a mut T
where
    T: Container,
{
    type Contract = <T as Container>::Contract;
}

pub trait Reborrow {
    type Target;

    fn reborrow(&self) -> &Self::Target;
}

pub trait ReborrowMut: Reborrow {
    fn reborrow_mut(&mut self) -> &mut Self::Target;
}

impl<'a, T> Reborrow for &'a T {
    type Target = T;

    fn reborrow(&self) -> &Self::Target {
        *self
    }
}

impl<'a, T> Reborrow for &'a mut T {
    type Target = T;

    fn reborrow(&self) -> &Self::Target {
        &**self
    }
}

impl<'a, T> ReborrowMut for &'a mut T {
    fn reborrow_mut(&mut self) -> &mut Self::Target {
        *self
    }
}

pub mod alias {
    use super::*;

    use graph::storage::Storage;
    use graph::topology::{Edge, Face, Vertex};

    pub type OwnedCore<G> = Core<Storage<Vertex<G>>, Storage<Edge<G>>, Storage<Face<G>>>;
}
