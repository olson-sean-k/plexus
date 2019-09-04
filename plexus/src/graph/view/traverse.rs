use std::collections::{HashSet, VecDeque};
use std::hash::Hash;
use std::marker::PhantomData;

use crate::graph::borrow::Reborrow;
use crate::graph::storage::AsStorage;
use crate::graph::view::{FromKeyedSource, IntoKeyedSource, IntoView, PayloadBinding, View};
use crate::graph::GraphError;

pub type BreadthTraversal<T, M> = Traversal<VecDeque<<T as PayloadBinding>::Key>, T, M>;
pub type DepthTraversal<T, M> = Traversal<Vec<<T as PayloadBinding>::Key>, T, M>;

/// Expresses adjacency of like topology.
///
/// View types that implement this trait provide some notion of _adjacency_,
/// where some topology has neighboring topology of the same type. For example,
/// vertices are connected to neighbors via arcs.
pub trait Adjacency: PayloadBinding {
    type Output: IntoIterator<Item = Self::Key>;

    /// Gets the keys of neighboring topology.
    fn adjacency(&self) -> Self::Output;
}

/// Linear buffer used for graph traversals.
///
/// The ordering of pushes and pops determines the ordering of a graph
/// traversal using the buffer.
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
    T: PayloadBinding,
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
    T: PayloadBinding,
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
    T: Into<View<M, <T as PayloadBinding>::Payload>> + PayloadBinding,
    M: Reborrow,
    M::Target: AsStorage<T::Payload>,
{
    fn from(view: T) -> Self {
        let (key, storage) = view.into().into_keyed_source();
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
    B: TraversalBuffer<<T as PayloadBinding>::Key>,
    T: Adjacency + Copy + FromKeyedSource<(<T as PayloadBinding>::Key, &'a M)>,
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

/// Trace of a graph traversal path.
///
/// Traversal traces typically determine when a traversal should terminate.
pub trait TraversalTrace<T> {
    // TODO: How useful is `Result` here? When used in iterators, there aren't
    //       many choices for handling errors.
    /// Visits a topology.
    ///
    /// This function accepts a _breadcrumb_ representing the topology, which
    /// is examined and may be stored. If the breadcrumb has been visited
    /// before (or the traversal should otherwise be terminated), then this
    /// function returns `true`, otherwise `false`.
    ///
    /// # Errors
    ///
    /// Returns an error if an unexpected breadcrumb is encountered. The
    /// semantics of this error depend on the implementation. Some
    /// implementations may never return an error.
    fn visit(&mut self, breadcrumb: T) -> Result<bool, GraphError>;
}

/// Trace that detects the first breadcrumb that is visited.
///
/// This trace only stores the first breadcrumb in a traversal and should
/// **not** be used when traversing a graph with unknown consistency.
#[derive(Clone, Copy, Debug, Default)]
pub struct TraceFirst<T>
where
    T: Copy,
{
    breadcrumb: Option<T>,
}

impl<T> TraversalTrace<T> for TraceFirst<T>
where
    T: Copy + Eq,
{
    fn visit(&mut self, breadcrumb: T) -> Result<bool, GraphError> {
        Ok(match self.breadcrumb {
            Some(intersection) => intersection == breadcrumb,
            None => {
                self.breadcrumb = Some(breadcrumb);
                false
            }
        })
    }
}

/// Trace that detects any breadcrumb that has been visited.
///
/// This trace stores all breadcrumbs and detects any and all collisions.
/// Additionally, a `TraceFirst` is used to detect premature collisions, which
/// result in a `TopologyMalformed` error.
#[derive(Clone, Debug, Default)]
pub struct TraceAny<T>
where
    T: Copy + Eq + Hash,
{
    trace: TraceFirst<T>,
    breadcrumbs: HashSet<T>,
}

impl<T> TraversalTrace<T> for TraceAny<T>
where
    T: Copy + Eq + Hash,
{
    fn visit(&mut self, breadcrumb: T) -> Result<bool, GraphError> {
        let expected = self.trace.visit(breadcrumb)?;
        if self.breadcrumbs.insert(breadcrumb) {
            Ok(false)
        }
        else {
            if expected {
                Ok(true)
            }
            else {
                Err(GraphError::TopologyMalformed)
            }
        }
    }
}
