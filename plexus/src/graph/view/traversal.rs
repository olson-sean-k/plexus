use std::collections::{HashSet, VecDeque};
use std::marker::PhantomData;

use crate::graph::borrow::Reborrow;
use crate::graph::storage::key::OpaqueKey;
use crate::graph::storage::AsStorage;
use crate::graph::view::{FromKeyedSource, IntoKeyedSource, IntoView, ViewBinding};

pub type BreadthTraversal<T, M> = Traversal<VecDeque<<T as ViewBinding<M>>::Key>, T, M>;
pub type DepthTraversal<T, M> = Traversal<Vec<<T as ViewBinding<M>>::Key>, T, M>;

pub trait Adjacency: Sized {
    type Output: IntoIterator<Item = Self::Key>;
    type Key: OpaqueKey;

    fn adjacency(&self) -> Self::Output;
}

pub trait TraversalBuffer<T>: Default + Extend<T> {
    fn push(&mut self, item: T);
    fn pop(&mut self) -> Option<T>;
}

impl<T> TraversalBuffer<T> for Vec<T> {
    fn push(&mut self, item: T) {
        Vec::<T>::push(self, item)
    }

    fn pop(&mut self) -> Option<T> {
        Vec::<T>::pop(self)
    }
}

impl<T> TraversalBuffer<T> for VecDeque<T> {
    fn push(&mut self, item: T) {
        VecDeque::<T>::push_back(self, item)
    }

    fn pop(&mut self) -> Option<T> {
        VecDeque::<T>::pop_front(self)
    }
}

pub struct Traversal<B, T, M>
where
    B: TraversalBuffer<T::Key>,
    T: ViewBinding<M>,
    M: Reborrow,
    M::Target: AsStorage<T::Payload>,
{
    storage: M,
    breadcrumbs: HashSet<T::Key>,
    buffer: B,
    phantom: PhantomData<T>,
}

impl<B, T, M> Clone for Traversal<B, T, M>
where
    B: Clone + TraversalBuffer<T::Key>,
    T: ViewBinding<M>,
    M: Clone + Reborrow,
    M::Target: AsStorage<T::Payload>,
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

impl<B, T, M> From<T> for Traversal<B, T, M>
where
    B: TraversalBuffer<T::Key>,
    T: ViewBinding<M>,
    M: Reborrow,
    M::Target: AsStorage<T::Payload>,
{
    fn from(view: T) -> Self {
        let (key, storage) = view.into_inner().into_keyed_source();
        let capacity = storage.reborrow().as_storage().len();
        let mut buffer = B::default();
        buffer.push(key);
        Traversal {
            storage,
            breadcrumbs: HashSet::with_capacity(capacity),
            buffer,
            phantom: PhantomData,
        }
    }
}

impl<'a, B, T, M> Iterator for Traversal<B, T, &'a M>
where
    B: TraversalBuffer<<T as Adjacency>::Key>,
    T: Adjacency
        + Copy
        + FromKeyedSource<(<T as Adjacency>::Key, &'a M)>
        + ViewBinding<&'a M, Key = <T as Adjacency>::Key>,
    M: 'a + AsStorage<T::Payload>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(view) = self
            .buffer
            .pop()
            .and_then(|key| -> Option<T> { (key, self.storage).into_view() })
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
            Some(AsStorage::<T::Payload>::as_storage(&self.storage).len()),
        )
    }
}
