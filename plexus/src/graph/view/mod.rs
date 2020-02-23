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
}

pub trait Rebind: ClosedView {
    // TODO: Simplify the bounds on this function. It may be necessary to
    //       implement it per type.
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
    fn rebind<T, B>(self, key: T::Key) -> Result<T, GraphError>
    where
        Self: Into<View<B, <Self as ClosedView>::Entity>>,
        T: From<View<B, <T as ClosedView>::Entity>> + ClosedView,
        B: Reborrow,
        B::Target: AsStorage<Self::Entity> + AsStorage<T::Entity>,
    {
        self.into()
            .rebind_into::<_, T::Entity>(key)
            .ok_or_else(|| GraphError::TopologyNotFound)
    }
}

// TODO: Remove this blanket implementation.
impl<T> Rebind for T where T: ClosedView {}

pub struct View<B, E>
where
    B: Reborrow,
    B::Target: AsStorage<E>,
    E: Entity,
{
    storage: B,
    key: E::Key,
}

impl<B, E> View<B, E>
where
    B: Reborrow,
    B::Target: AsStorage<E>,
    E: Entity,
{
    pub fn bind(storage: B, key: E::Key) -> Option<Self> {
        storage
            .reborrow()
            .as_storage()
            .contains_key(&key)
            .some(View::bind_unchecked(storage, key))
    }

    pub fn bind_into<U>(storage: B, key: E::Key) -> Option<U>
    where
        U: From<Self>,
    {
        View::bind(storage, key).map(U::from)
    }

    pub fn unbind(self) -> (B, E::Key) {
        let View { storage, key, .. } = self;
        (storage, key)
    }

    pub fn rebind<U>(self, key: U::Key) -> Option<View<B, U>>
    where
        U: Entity,
        B::Target: AsStorage<U>,
    {
        let (storage, _) = self.unbind();
        View::bind(storage, key)
    }

    pub fn rebind_into<V, U>(self, key: U::Key) -> Option<V>
    where
        V: From<View<B, U>>,
        U: Entity,
        B::Target: AsStorage<U>,
    {
        self.rebind(key).map(V::from)
    }

    pub fn key(&self) -> E::Key {
        self.key
    }

    pub fn interior_reborrow(&self) -> View<&B::Target, E> {
        View::bind_unchecked(self.storage.reborrow(), self.key)
    }

    pub(in crate::graph) fn bind_unchecked(storage: B, key: E::Key) -> Self {
        View { storage, key }
    }
}

impl<B, E> View<B, E>
where
    B: ReborrowMut,
    B::Target: AsStorage<E>,
    E: Entity,
{
    pub fn interior_reborrow_mut(&mut self) -> View<&mut B::Target, E> {
        View::bind_unchecked(self.storage.reborrow_mut(), self.key)
    }
}

impl<'a, M, E> View<&'a mut M, E>
where
    M: 'a + AsStorageMut<E>,
    E: 'a + Entity,
{
    pub fn into_ref(self) -> View<&'a M, E> {
        let (storage, key) = self.unbind();
        View::bind(&*storage, key).unwrap()
    }
}

impl<B, E> Clone for View<B, E>
where
    B: Clone + Reborrow,
    B::Target: AsStorage<E>,
    E: Entity,
{
    fn clone(&self) -> Self {
        View {
            storage: self.storage.clone(),
            key: self.key,
        }
    }
}

impl<B, E> Copy for View<B, E>
where
    B: Copy + Reborrow,
    B::Target: AsStorage<E>,
    E: Entity,
{
}

impl<B, E> Deref for View<B, E>
where
    B: Reborrow,
    B::Target: AsStorage<E>,
    E: Entity,
{
    type Target = E;

    fn deref(&self) -> &Self::Target {
        self.storage
            .reborrow()
            .as_storage()
            .get(&self.key)
            .expect("view key invalidated")
    }
}

impl<B, E> DerefMut for View<B, E>
where
    B: ReborrowMut,
    B::Target: AsStorageMut<E>,
    E: Entity,
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
impl<B, E> PartialEq for View<B, E>
where
    B: Reborrow,
    B::Target: AsStorage<E> + Consistent,
    E: Entity,
{
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

pub struct Orphan<'a, E>
where
    E: Entity,
{
    payload: &'a mut E,
    key: E::Key,
}

impl<'a, E> Orphan<'a, E>
where
    E: 'a + Entity,
{
    pub fn bind<M>(storage: &'a mut M, key: E::Key) -> Option<Self>
    where
        M: AsStorageMut<E>,
    {
        View::bind(storage, key).map(Orphan::from)
    }

    pub fn bind_into<U, M>(storage: &'a mut M, key: E::Key) -> Option<U>
    where
        U: From<Self>,
        M: AsStorageMut<E>,
    {
        Orphan::bind(storage, key).map(U::from)
    }

    pub fn key(&self) -> E::Key {
        self.key
    }

    pub(in crate::graph) fn bind_unchecked(payload: &'a mut E, key: E::Key) -> Self {
        Orphan { payload, key }
    }
}

impl<'a, E> Deref for Orphan<'a, E>
where
    E: 'a + Entity,
{
    type Target = E;

    fn deref(&self) -> &Self::Target {
        &*self.payload
    }
}

impl<'a, E> DerefMut for Orphan<'a, E>
where
    E: 'a + Entity,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.payload
    }
}

impl<'a, E, M> From<View<&'a mut M, E>> for Orphan<'a, E>
where
    E: 'a + Entity,
    M: AsStorageMut<E>,
{
    fn from(view: View<&'a mut M, E>) -> Self {
        let (storage, key) = view.unbind();
        let payload = storage
            .as_storage_mut()
            .get_mut(&key)
            .expect("view key invalidated");
        Orphan::bind_unchecked(payload, key)
    }
}
