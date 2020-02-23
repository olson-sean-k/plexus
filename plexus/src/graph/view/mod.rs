pub mod edge;
pub mod face;
pub mod path;
mod traverse;
pub mod vertex;

use fool::BoolExt;
use std::ops::{Deref, DerefMut};

use crate::graph::borrow::{Reborrow, ReborrowMut};
use crate::graph::mutation::Consistent;
use crate::graph::storage::key::OpaqueKey;
use crate::graph::storage::payload::Entity;
use crate::graph::storage::{AsStorage, AsStorageMut};
use crate::graph::GraphError;

// TODO: Use `bind_unchecked` whenever possible (that is, when it is logically
//       consistent to assume that the key is present in storage).
// TODO: Consider `Bind` and `Unbind` traits and decomposing the `ClosedView`
//       trait.

pub trait ClosedView: Deref<Target = <Self as ClosedView>::Entity> {
    type Key: OpaqueKey;
    type Entity: Entity<Key = Self::Key>;

    /// Gets the key for the view.
    fn key(&self) -> Self::Key;

    /// Rebinds a view's storage with the given key.
    ///
    /// Rebinding a view allows its underlying storage to be reinterpretted.
    /// The output view must also be bound to a payload.
    ///
    /// # Examples
    ///
    /// Perform a fallible traversal and preserve mutability of the resulting
    /// view:
    ///
    /// ```rust,no_run
    /// # use plexus::graph::{ClosedView, MeshGraph};
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
    ///         face.rebind(key).unwrap() // Rebind into the boundary face.
    ///     }
    ///     else {
    ///         face
    ///     }
    /// };
    /// ```
    fn rebind<T, M>(self, key: T::Key) -> Result<T, GraphError>
    where
        Self: Into<View<M, <Self as ClosedView>::Entity>>,
        T: From<View<M, <T as ClosedView>::Entity>> + ClosedView,
        M: Reborrow,
        M::Target: AsStorage<Self::Entity> + AsStorage<T::Entity>,
    {
        self.into()
            .rebind_into::<_, T::Entity>(key)
            .ok_or_else(|| GraphError::TopologyNotFound)
    }
}

pub struct View<M, T>
where
    M: Reborrow,
    M::Target: AsStorage<T>,
    T: Entity,
{
    storage: M,
    key: T::Key,
}

impl<M, T> View<M, T>
where
    M: Reborrow,
    M::Target: AsStorage<T>,
    T: Entity,
{
    pub fn bind(storage: M, key: T::Key) -> Option<Self> {
        storage
            .reborrow()
            .as_storage()
            .contains_key(&key)
            .some(View::bind_unchecked(storage, key))
    }

    pub fn bind_into<U>(storage: M, key: T::Key) -> Option<U>
    where
        U: From<Self>,
    {
        View::bind(storage, key).map(U::from)
    }

    pub fn unbind(self) -> (M, T::Key) {
        let View { storage, key, .. } = self;
        (storage, key)
    }

    pub fn rebind<U>(self, key: U::Key) -> Option<View<M, U>>
    where
        U: Entity,
        M::Target: AsStorage<U>,
    {
        let (storage, _) = self.unbind();
        View::bind(storage, key)
    }

    pub fn rebind_into<V, U>(self, key: U::Key) -> Option<V>
    where
        V: From<View<M, U>>,
        U: Entity,
        M::Target: AsStorage<U>,
    {
        self.rebind(key).map(V::from)
    }

    pub fn key(&self) -> T::Key {
        self.key
    }

    pub fn interior_reborrow(&self) -> View<&M::Target, T> {
        View::bind_unchecked(self.storage.reborrow(), self.key)
    }

    pub(in crate::graph) fn bind_unchecked(storage: M, key: T::Key) -> Self {
        View { storage, key }
    }
}

impl<M, T> View<M, T>
where
    M: ReborrowMut,
    M::Target: AsStorage<T>,
    T: Entity,
{
    pub fn interior_reborrow_mut(&mut self) -> View<&mut M::Target, T> {
        View::bind_unchecked(self.storage.reborrow_mut(), self.key)
    }
}

impl<'a, M, T> View<&'a mut M, T>
where
    M: 'a + AsStorageMut<T>,
    T: 'a + Entity,
{
    pub fn into_ref(self) -> View<&'a M, T> {
        let (storage, key) = self.unbind();
        View::bind(&*storage, key).unwrap()
    }
}

impl<M, T> Clone for View<M, T>
where
    M: Clone + Reborrow,
    M::Target: AsStorage<T>,
    T: Entity,
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
    T: Entity,
{
}

impl<M, T> Deref for View<M, T>
where
    M: Reborrow,
    M::Target: AsStorage<T>,
    T: Entity,
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
    M: ReborrowMut,
    M::Target: AsStorageMut<T>,
    T: Entity,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.storage
            .reborrow_mut()
            .as_storage_mut()
            .get_mut(&self.key)
            .expect("view key invalidated")
    }
}

// TODO: Consider implementing `Eq` for views.
impl<M, T> PartialEq for View<M, T>
where
    M: Reborrow,
    M::Target: AsStorage<T> + Consistent,
    T: Entity,
{
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

pub struct Orphan<'a, T>
where
    T: Entity,
{
    payload: &'a mut T,
    key: T::Key,
}

impl<'a, T> Orphan<'a, T>
where
    T: 'a + Entity,
{
    pub fn bind<M>(storage: &'a mut M, key: T::Key) -> Option<Self>
    where
        M: AsStorageMut<T>,
    {
        View::bind(storage, key).map(Orphan::from)
    }

    pub fn bind_into<U, M>(storage: &'a mut M, key: T::Key) -> Option<U>
    where
        U: From<Self>,
        M: AsStorageMut<T>,
    {
        Orphan::bind(storage, key).map(U::from)
    }

    pub fn key(&self) -> T::Key {
        self.key
    }

    pub(in crate::graph) fn bind_unchecked(payload: &'a mut T, key: T::Key) -> Self {
        Orphan { payload, key }
    }
}

impl<'a, T> Deref for Orphan<'a, T>
where
    T: 'a + Entity,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.payload
    }
}

impl<'a, T> DerefMut for Orphan<'a, T>
where
    T: 'a + Entity,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.payload
    }
}

impl<'a, T, M> From<View<&'a mut M, T>> for Orphan<'a, T>
where
    T: 'a + Entity,
    M: AsStorageMut<T>,
{
    fn from(view: View<&'a mut M, T>) -> Self {
        let (storage, key) = view.unbind();
        let payload = storage
            .as_storage_mut()
            .get_mut(&key)
            .expect("view key invalidated");
        Orphan::bind_unchecked(payload, key)
    }
}
