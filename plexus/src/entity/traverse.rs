use std::collections::{HashSet, VecDeque};
use std::marker::PhantomData;

use crate::entity::borrow::Reborrow;
use crate::entity::storage::AsStorage;
use crate::entity::view::{Bind, ClosedView, Unbind};

pub enum Breadth {}
pub enum Depth {}

pub trait Order<T>
where
    T: Adjacency,
{
    type Buffer: Buffer<T::Key>;
}

impl<T> Order<T> for Breadth
where
    T: Adjacency,
{
    type Buffer = VecDeque<T::Key>;
}

impl<T> Order<T> for Depth
where
    T: Adjacency,
{
    type Buffer = Vec<T::Key>;
}

pub trait Adjacency: ClosedView {
    type Output: IntoIterator<Item = Self::Key>;

    fn adjacency(&self) -> Self::Output;
}

pub trait Buffer<T>: Default + Extend<T> {
    fn push(&mut self, item: T);
    fn pop(&mut self) -> Option<T>;
}

impl<T> Buffer<T> for Vec<T> {
    fn push(&mut self, item: T) {
        Vec::<T>::push(self, item)
    }

    fn pop(&mut self) -> Option<T> {
        Vec::<T>::pop(self)
    }
}

impl<T> Buffer<T> for VecDeque<T> {
    fn push(&mut self, item: T) {
        VecDeque::<T>::push_back(self, item)
    }

    fn pop(&mut self) -> Option<T> {
        VecDeque::<T>::pop_front(self)
    }
}

#[derive(Debug)]
pub struct Traversal<B, T, R = Depth>
where
    B: Reborrow,
    B::Target: AsStorage<T::Entity>,
    T: Adjacency,
    R: Order<T>,
{
    storage: B,
    breadcrumbs: HashSet<T::Key>,
    buffer: R::Buffer,
    phantom: PhantomData<T>,
}

impl<B, T, R> Clone for Traversal<B, T, R>
where
    B: Clone + Reborrow,
    B::Target: AsStorage<T::Entity>,
    T: Adjacency,
    R: Order<T>,
    R::Buffer: Clone,
{
    fn clone(&self) -> Self {
        Traversal {
            storage: self.storage.clone(),
            breadcrumbs: self.breadcrumbs.clone(),
            buffer: self.buffer.clone(),
            phantom: PhantomData,
        }
    }
}

impl<B, T, R> From<T> for Traversal<B, T, R>
where
    B: Reborrow,
    B::Target: AsStorage<T::Entity>,
    T: Adjacency + Unbind<B>,
    R: Order<T>,
{
    fn from(view: T) -> Self {
        let (storage, key) = view.unbind();
        let capacity = storage.reborrow().as_storage().len();
        let mut buffer = R::Buffer::default();
        buffer.push(key);
        Traversal {
            storage,
            breadcrumbs: HashSet::with_capacity(capacity),
            buffer,
            phantom: PhantomData,
        }
    }
}

impl<'a, M, T, R> Iterator for Traversal<&'a M, T, R>
where
    M: 'a + AsStorage<T::Entity>,
    T: Adjacency + Bind<&'a M>,
    R: Order<T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(view) = self.buffer.pop().and_then(|key| T::bind(self.storage, key)) {
            if self.breadcrumbs.insert(view.key()) {
                self.buffer.extend(view.adjacency());
                return Some(view);
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            1,
            Some(AsStorage::<T::Entity>::as_storage(&self.storage).len()),
        )
    }
}
