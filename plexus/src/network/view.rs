use fool::BoolExt;
use std::ops::{Deref, DerefMut};

use crate::network::borrow::{Reborrow, ReborrowMut};
use crate::network::storage::{AsStorage, AsStorageMut, OpaqueKey};
use crate::network::Entity;

pub trait ClosedView: Deref<Target = <Self as ClosedView>::Entity> {
    type Key: OpaqueKey;
    type Entity: Entity<Key = Self::Key>;

    fn key(&self) -> Self::Key;
}

pub trait Rebind<T>: ClosedView
where
    T: ClosedView,
{
    fn rebind(self, key: T::Key) -> Option<T>;
}

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

    pub fn bind_unchecked(storage: B, key: E::Key) -> Self {
        View { storage, key }
    }

    pub fn bind_into<T>(storage: B, key: E::Key) -> Option<T>
    where
        T: From<Self>,
    {
        View::bind(storage, key).map(T::from)
    }

    pub fn unbind(self) -> (B, E::Key) {
        let View { storage, key, .. } = self;
        (storage, key)
    }

    pub fn rebind<T>(self, key: T::Key) -> Option<View<B, T>>
    where
        B::Target: AsStorage<T>,
        T: Entity,
    {
        let (storage, _) = self.unbind();
        View::bind(storage, key)
    }

    pub fn rebind_into<T, U>(self, key: U::Key) -> Option<T>
    where
        B::Target: AsStorage<U>,
        T: From<View<B, U>>,
        U: Entity,
    {
        self.rebind(key).map(T::from)
    }

    pub fn key(&self) -> E::Key {
        self.key
    }

    pub fn interior_reborrow(&self) -> View<&B::Target, E> {
        View::bind_unchecked(self.storage.reborrow(), self.key)
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
    pub fn to_ref(&self) -> View<&M, E> {
        View::bind(&*self.storage, self.key).unwrap()
    }

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

impl<B, E> ClosedView for View<B, E>
where
    B: Reborrow,
    B::Target: AsStorage<E>,
    E: Entity,
{
    type Key = E::Key;
    type Entity = E;

    fn key(&self) -> Self::Key {
        self.key
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
    B::Target: AsStorage<E>,
    E: Entity,
{
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl<B, E, T> Rebind<T> for View<B, E>
where
    B: Reborrow,
    B::Target: AsStorage<E> + AsStorage<T::Entity>,
    E: Entity,
    T: ClosedView + From<View<B, <T as ClosedView>::Entity>>,
{
    fn rebind(self, key: T::Key) -> Option<T> {
        self.rebind(key).map(T::from)
    }
}

pub struct Orphan<'a, E>
where
    E: Entity,
{
    entity: &'a mut E,
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

    pub fn bind_unchecked(entity: &'a mut E, key: E::Key) -> Self {
        Orphan { entity, key }
    }

    pub fn bind_into<T, M>(storage: &'a mut M, key: E::Key) -> Option<T>
    where
        T: From<Self>,
        M: AsStorageMut<E>,
    {
        Orphan::bind(storage, key).map(T::from)
    }

    pub fn key(&self) -> E::Key {
        self.key
    }
}

impl<'a, E> ClosedView for Orphan<'a, E>
where
    E: 'a + Entity,
{
    type Key = E::Key;
    type Entity = E;

    fn key(&self) -> Self::Key {
        self.key
    }
}

impl<'a, E> Deref for Orphan<'a, E>
where
    E: 'a + Entity,
{
    type Target = E;

    fn deref(&self) -> &Self::Target {
        &*self.entity
    }
}

impl<'a, E> DerefMut for Orphan<'a, E>
where
    E: 'a + Entity,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.entity
    }
}

impl<'a, E, M> From<View<&'a mut M, E>> for Orphan<'a, E>
where
    E: 'a + Entity,
    M: AsStorageMut<E>,
{
    fn from(view: View<&'a mut M, E>) -> Self {
        let (storage, key) = view.unbind();
        let entity = storage
            .as_storage_mut()
            .get_mut(&key)
            .expect("view key invalidated");
        Orphan::bind_unchecked(entity, key)
    }
}
