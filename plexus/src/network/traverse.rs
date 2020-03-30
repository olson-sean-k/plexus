use std::collections::{HashSet, VecDeque};
use std::marker::PhantomData;

use crate::network::borrow::Reborrow;
use crate::network::storage::AsStorage;
use crate::network::view::{ClosedView, View};

pub type BreadthTraversal<B, T> = Traversal<B, T, VecDeque<<T as ClosedView>::Key>>;
pub type DepthTraversal<B, T> = Traversal<B, T, Vec<<T as ClosedView>::Key>>;

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
pub struct Traversal<B, T, R>
where
    B: Reborrow,
    B::Target: AsStorage<T::Entity>,
    T: ClosedView,
    R: Buffer<T::Key>,
{
    storage: B,
    breadcrumbs: HashSet<T::Key>,
    buffer: R,
    phantom: PhantomData<T>,
}

impl<B, T, R> Clone for Traversal<B, T, R>
where
    B: Clone + Reborrow,
    B::Target: AsStorage<T::Entity>,
    T: ClosedView,
    R: Clone + Buffer<T::Key>,
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
    T: Into<View<B, <T as ClosedView>::Entity>> + ClosedView,
    R: Buffer<T::Key>,
{
    fn from(view: T) -> Self {
        let (storage, key) = view.into().unbind();
        let capacity = storage.reborrow().as_storage().len();
        let mut buffer = R::default();
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
    T: Adjacency + Copy + From<View<&'a M, <T as ClosedView>::Entity>>,
    R: Buffer<<T as ClosedView>::Key>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(view) = self
            .buffer
            .pop()
            .and_then(|key| -> Option<T> { View::bind_into(self.storage, key) })
        {
            if self.breadcrumbs.insert(view.key()) {
                self.buffer.extend(view.adjacency());
                return Some(view);
            }
            else {
                continue;
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
