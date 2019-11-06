pub mod edge;
pub mod face;
pub mod path;
mod traverse;
pub mod vertex;

use fool::BoolExt as _;
use std::ops::{Deref, DerefMut};

use crate::graph::borrow::{Reborrow, ReborrowMut};
use crate::graph::core::Bind;
use crate::graph::mutation::Consistent;
use crate::graph::storage::key::OpaqueKey;
use crate::graph::storage::payload::Payload;
use crate::graph::storage::{AsStorage, AsStorageMut};
use crate::graph::GraphError;

pub trait IntoKeyedSource<T>: Sized {
    fn into_keyed_source(self) -> T;
}

pub trait FromKeyedSource<T>: Sized {
    fn from_keyed_source(source: T) -> Option<Self>;
}

pub trait IntoView<T>: Sized {
    fn into_view(self) -> Option<T>;
}

impl<T, U> IntoView<U> for T
where
    T: Sized,
    U: FromKeyedSource<T>,
{
    fn into_view(self) -> Option<U> {
        U::from_keyed_source(self)
    }
}

/// A view into a graph that is bound to a payload by its key.
///
/// This trait is implemented by views that operate over a specific topology in
/// a graph, such as a vertex or face. Note that rings and paths are exposed as
/// _meta-views_, and as such do not implement this trait.
pub trait PayloadBinding: Deref<Target = <Self as PayloadBinding>::Payload> {
    // This associated type is redundant, but avoids re-exporting the
    // `Payload` trait and simplifies the use of this trait.
    type Key: OpaqueKey;
    type Payload: Payload<Key = Self::Key>;

    /// Gets the key for the view.
    fn key(&self) -> Self::Key;

    /// Rekeys a view into another.
    ///
    /// Rekeying a view allows its underlying storage to be reinterpretted. The
    /// output view must also be bound to a payload.
    ///
    /// # Examples
    ///
    /// Perform a fallible traversal and preserve mutability of the resulting
    /// view:
    ///
    /// ```rust,no_run
    /// # use plexus::graph::{PayloadBinding, MeshGraph};
    /// # use plexus::prelude::*;
    /// #
    /// # let mut graph = MeshGraph::<()>::default();
    /// # let key = graph.faces().keys().nth(0).unwrap();
    /// // ...
    /// let face = graph.face_mut(key).unwrap();
    /// // Find a face along a boundary. If no such face is found, continue to use the
    /// // initiating face.
    /// let mut face = {
    ///     let key = face
    ///         .traverse_by_depth()
    ///         .find(|face| {
    ///             face.interior_arcs()
    ///                 .map(|arc| arc.into_opposite_arc())
    ///                 .any(|arc| arc.is_boundary_arc())
    ///         })
    ///         .map(|face| face.key());
    ///     if let Some(key) = key {
    ///         face.rekey(key).unwrap() // Rekey into the boundary face.
    ///     }
    ///     else {
    ///         face
    ///     }
    /// };
    /// ```
    fn rekey<T, M>(self, key: T::Key) -> Result<T, GraphError>
    where
        Self: Into<View<M, <Self as PayloadBinding>::Payload>>,
        T: From<View<M, <T as PayloadBinding>::Payload>> + PayloadBinding,
        M: Reborrow,
        M::Target: AsStorage<Self::Payload> + AsStorage<T::Payload>,
    {
        self.into()
            .rekey_map::<_, T::Payload>(key)
            .ok_or_else(|| GraphError::TopologyNotFound)
    }
}

pub struct View<M, T>
where
    M: Reborrow,
    M::Target: AsStorage<T>,
    T: Payload,
{
    key: T::Key,
    storage: M,
}

impl<M, T> View<M, T>
where
    M: Reborrow,
    M::Target: AsStorage<T>,
    T: Payload,
{
    fn from_keyed_source_unchecked(source: (T::Key, M)) -> Self {
        let (key, storage) = source;
        View { key, storage }
    }

    pub fn key(&self) -> T::Key {
        self.key
    }

    pub fn bind<U, N>(self, storage: N) -> View<<M as Bind<U, N>>::Output, T>
    where
        U: Payload,
        N: AsStorage<U>,
        M: Bind<U, N>,
        M::Output: Reborrow,
        <M::Output as Reborrow>::Target: AsStorage<T>,
    {
        let (key, source) = self.into_keyed_source();
        View::from_keyed_source_unchecked((key, source.bind(storage)))
    }

    pub fn rekey<U>(self, key: U::Key) -> Option<View<M, U>>
    where
        U: Payload,
        M::Target: AsStorage<U>,
    {
        let (_, storage) = self.into_keyed_source();
        (key, storage).into_view()
    }

    pub fn rekey_map<V, U>(self, key: U::Key) -> Option<V>
    where
        V: From<View<M, U>>,
        U: Payload,
        M::Target: AsStorage<U>,
    {
        self.rekey(key).map(V::from)
    }

    pub fn interior_reborrow(&self) -> View<&M::Target, T> {
        View::from_keyed_source_unchecked((self.key, self.storage.reborrow()))
    }
}

impl<M, T> View<M, T>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<T>,
    T: Payload,
{
    pub fn interior_reborrow_mut(&mut self) -> View<&mut M::Target, T> {
        View::from_keyed_source_unchecked((self.key, self.storage.reborrow_mut()))
    }
}

impl<'a, M, T> View<&'a mut M, T>
where
    M: 'a + AsStorage<T> + AsStorageMut<T>,
    T: 'a + Payload,
{
    pub fn into_orphan(self) -> Orphan<'a, T> {
        let (key, storage) = self.into_keyed_source();
        (key, storage).into_view().unwrap()
    }

    pub fn into_ref(self) -> View<&'a M, T> {
        let (key, storage) = self.into_keyed_source();
        (key, &*storage).into_view().unwrap()
    }
}

impl<M, T> Clone for View<M, T>
where
    M: Clone + Reborrow,
    M::Target: AsStorage<T>,
    T: Payload,
{
    fn clone(&self) -> Self {
        View {
            storage: self.storage.clone(),
            key: self.key,
        }
    }
}

impl<M, T> Copy for View<M, T>
where
    M: Copy + Reborrow,
    M::Target: AsStorage<T>,
    T: Payload,
{
}

impl<M, T> Deref for View<M, T>
where
    M: Reborrow,
    M::Target: AsStorage<T>,
    T: Payload,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.storage
            .reborrow()
            .as_storage()
            .get(&self.key)
            .expect("view key invalidated")
    }
}

impl<M, T> DerefMut for View<M, T>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<T> + AsStorageMut<T>,
    T: Payload,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.storage
            .reborrow_mut()
            .as_storage_mut()
            .get_mut(&self.key)
            .expect("view key invalidated")
    }
}

impl<M, T> FromKeyedSource<(T::Key, M)> for View<M, T>
where
    M: Reborrow,
    M::Target: AsStorage<T>,
    T: Payload,
{
    fn from_keyed_source(source: (T::Key, M)) -> Option<Self> {
        let (key, storage) = source;
        storage
            .reborrow()
            .as_storage()
            .contains_key(&key)
            .some(View::from_keyed_source_unchecked((key, storage)))
    }
}

impl<M, T> IntoKeyedSource<(T::Key, M)> for View<M, T>
where
    M: Reborrow,
    M::Target: AsStorage<T>,
    T: Payload,
{
    fn into_keyed_source(self) -> (T::Key, M) {
        let View { key, storage, .. } = self;
        (key, storage)
    }
}

// TODO: Consider implementing `Eq` for views.
impl<M, T> PartialEq for View<M, T>
where
    M: Reborrow,
    M::Target: AsStorage<T> + Consistent,
    T: Payload,
{
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

pub struct Orphan<'a, T>
where
    T: Payload,
{
    key: T::Key,
    payload: &'a mut T,
}

impl<'a, T> Orphan<'a, T>
where
    T: 'a + Payload,
{
    pub fn from_keyed_source_unchecked(source: (T::Key, &'a mut T)) -> Self {
        let (key, payload) = source;
        Orphan { key, payload }
    }

    pub fn key(&self) -> T::Key {
        self.key
    }
}

impl<'a, T> Deref for Orphan<'a, T>
where
    T: 'a + Payload,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.payload
    }
}

impl<'a, T> DerefMut for Orphan<'a, T>
where
    T: 'a + Payload,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.payload
    }
}

impl<'a, M, T> FromKeyedSource<(T::Key, &'a mut M)> for Orphan<'a, T>
where
    M: AsStorage<T> + AsStorageMut<T>,
    T: 'a + Payload,
{
    fn from_keyed_source(source: (T::Key, &'a mut M)) -> Option<Self> {
        let (key, storage) = source;
        storage
            .as_storage_mut()
            .get_mut(&key)
            .map(|payload| Orphan { key, payload })
    }
}
