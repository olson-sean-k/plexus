use std::collections::{HashSet, VecDeque};
use std::hash::Hash;
use std::marker::PhantomData;

use crate::entity::borrow::Reborrow;
use crate::entity::storage::{AsStorage, Enumerate};
use crate::entity::view::{Bind, ClosedView, Unbind};

pub enum Breadth {}
pub enum Depth {}

pub trait Buffer<R, T>: Default + Extend<T>
where
    R: Order<T>,
{
    fn push(&mut self, item: T);
    fn pop(&mut self) -> Option<T>;
}

impl<T> Buffer<Depth, T> for Vec<T> {
    fn push(&mut self, item: T) {
        Vec::<T>::push(self, item)
    }

    fn pop(&mut self) -> Option<T> {
        Vec::<T>::pop(self)
    }
}

impl<T> Buffer<Breadth, T> for VecDeque<T> {
    fn push(&mut self, item: T) {
        VecDeque::<T>::push_back(self, item)
    }

    fn pop(&mut self) -> Option<T> {
        VecDeque::<T>::pop_front(self)
    }
}

/// Traversal ordering.
///
/// Provides a default type implementing [`Buffer`] for the ordering described
/// by `Self`. This reduces the number of required type parameters in types
/// implementing traversals, as only an ordering type is needed to derive the
/// buffer. Note that the item type can typically be derived from other required
/// type parameters.
///
/// See [`Traversal`].
///
/// [`Buffer`]: crate::entity::traverse::Buffer
/// [`Traversal`]: crate::entity::traverse::Traversal
pub trait Order<T>: Sized {
    type Buffer: Buffer<Self, T>;
}

impl<T> Order<T> for Breadth {
    type Buffer = VecDeque<T>;
}

impl<T> Order<T> for Depth {
    type Buffer = Vec<T>;
}

pub trait Adjacency: ClosedView {
    type Output: IntoIterator<Item = Self::Key>;

    fn adjacency(&self) -> Self::Output;
}

#[derive(Debug)]
pub struct Traversal<B, T, R = Depth>
where
    B: Reborrow,
    B::Target: AsStorage<T::Entity>,
    T: Adjacency,
    R: Order<T::Key>,
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
    R: Order<T::Key>,
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
    R: Order<T::Key>,
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
    R: Order<T::Key>,
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

/// Trace of a traversal, iteration, etc.
///
/// A trace caches _breadcrumbs_, which identify entities encountered during a
/// traversal.
pub trait Trace<T> {
    /// Inserts the given breadcrumb into the trace. The breadcrumb may or may
    /// not be cached.
    ///
    /// A _collision_ occurs if a breadcrumb that has been cached by the trace
    /// is reinserted. If a collision with the trace is detected, then this
    /// function returns `false` and otherwise returns `true` (similarly to
    /// collections like [`HashSet`]).
    ///
    /// If `false` is returned, then any traversal or iteration should
    /// terminate.
    ///
    /// [`HashSet`]: std::collections::HashSet
    fn insert(&mut self, breadcrumb: T) -> bool;
}

/// Trace that caches and detects collisions with only the first breadcrumb that
/// is inserted.
///
/// A collision only occurs if the first breadcrumb is reinserted; no other
/// breadcrumbs are cached.
///
/// This trace should **not** be used when traversing a structure with unknown
/// consistency, because it may never signal that the iteration should
/// terminate. However, it requires very little space and time to operate.
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
            Some(collision) => collision != breadcrumb,
            None => {
                self.breadcrumb = Some(breadcrumb);
                true
            }
        }
    }
}

/// Trace that caches all inserted breadcrumbs and detects collisions with any
/// such breadcrumb.
///
/// This trace is very robust and reliably signals termination of a traversal,
/// but requires non-trivial space to cache breadcrumbs and must hash
/// breadcrumbs to detect collisions.
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
