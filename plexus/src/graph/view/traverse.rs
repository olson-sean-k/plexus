use std::collections::{HashSet, VecDeque};
use std::hash::Hash;
use std::marker::PhantomData;

use crate::graph::borrow::Reborrow;
use crate::graph::storage::AsStorage;
use crate::graph::view::{ClosedView, View};

pub type BreadthTraversal<T, M> = Traversal<VecDeque<<T as ClosedView>::Key>, T, M>;
pub type DepthTraversal<T, M> = Traversal<Vec<<T as ClosedView>::Key>, T, M>;

/// Expresses adjacency of like topology.
///
/// View types that implement this trait provide some notion of _adjacency_,
/// where some topology has neighboring topology of the same type. For example,
/// vertices are connected to neighbors via arcs.
pub trait Adjacency: ClosedView {
    type Output: IntoIterator<Item = Self::Key>;

    /// Gets the keys of neighboring topology.
    fn adjacency(&self) -> Self::Output;
}

/// Linear buffer used for graph traversals.
///
/// The ordering of pushes and pops determines the ordering of a graph traversal
/// using the buffer.
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

/// Graph traversal.
///
/// Traverses a graph and exposes views into the graph as an iterator. The
/// ordering of the traversal is determined by the `TraversalBuffer`.
///
/// See the `BreadthTraversal` and `DepthTraversal` type definitions.
#[derive(Debug)]
pub struct Traversal<B, T, M>
where
    B: TraversalBuffer<T::Key>,
    T: ClosedView,
    M: Reborrow,
    M::Target: AsStorage<T::Entity>,
{
    storage: M,
    breadcrumbs: HashSet<T::Key>,
    buffer: B,
    phantom: PhantomData<T>,
}

impl<B, T, M> Clone for Traversal<B, T, M>
where
    B: Clone + TraversalBuffer<T::Key>,
    T: ClosedView,
    M: Clone + Reborrow,
    M::Target: AsStorage<T::Entity>,
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
    T: Into<View<M, <T as ClosedView>::Entity>> + ClosedView,
    M: Reborrow,
    M::Target: AsStorage<T::Entity>,
{
    fn from(view: T) -> Self {
        let (storage, key) = view.into().unbind();
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
    B: TraversalBuffer<<T as ClosedView>::Key>,
    T: Adjacency + Copy + From<View<&'a M, <T as ClosedView>::Entity>>,
    M: 'a + AsStorage<T::Entity>,
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

/// Trace of a path.
pub trait Trace<T> {
    /// Inserts the given breadcrumb into the trace.
    ///
    /// If an intersection with the trace is detected, then this function
    /// returns `false` and otherwise returns `true` (similarly to collections
    /// like `HashSet). If `false` is returned, then the iteration should
    /// terminate.
    fn insert(&mut self, breadcrumb: T) -> bool;
}

/// Trace that detects the first breadcrumb that is encountered.
///
/// This trace only stores the first breadcrumb in a traversal and should
/// **not** be used when traversing a graph with unknown consistency, because it
/// may never signal that the iteration should terminate. However, it requires
/// very little space and time to operate.
#[derive(Clone, Copy, Debug, Default)]
pub struct TraceFirst<T>
where
    T: Copy,
{
    breadcrumb: Option<T>,
}

impl<T> Trace<T> for TraceFirst<T>
where
    T: Copy + Eq,
{
    fn insert(&mut self, breadcrumb: T) -> bool {
        match self.breadcrumb {
            Some(intersection) => intersection != breadcrumb,
            None => {
                self.breadcrumb = Some(breadcrumb);
                true
            }
        }
    }
}

/// Trace that detects any breadcrumb that has been previously encountered.
///
/// This trace stores all breadcrumbs and detects any and all collisions. This
/// is very robust, but requires space for breadcrumbs and must hash breadcrumbs
/// to detect collisions.
#[derive(Clone, Debug, Default)]
pub struct TraceAny<T>
where
    T: Copy + Eq + Hash,
{
    breadcrumbs: HashSet<T>,
}

impl<T> Trace<T> for TraceAny<T>
where
    T: Copy + Eq + Hash,
{
    fn insert(&mut self, breadcrumb: T) -> bool {
        self.breadcrumbs.insert(breadcrumb)
    }
}
