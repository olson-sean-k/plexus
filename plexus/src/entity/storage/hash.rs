use fnv::FnvBuildHasher;
use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

use crate::entity::storage::{
    AsStorage, AsStorageMut, DependentStorage, Dispatch, Dynamic, Enumerate, Get, InnerKey,
    InsertWithKey, Key, Mode, Remove, Static, StorageTarget,
};
use crate::entity::{Entity, Payload};

pub struct HashStorage<E, P = Static>
where
    E: Entity,
    P: Mode,
{
    inner: HashMap<InnerKey<<E as Entity>::Key>, E, FnvBuildHasher>,
    phantom: PhantomData<P>,
}

impl<E> AsStorage<E> for HashStorage<E, Dynamic>
where
    E: Entity<Storage = Self>,
    InnerKey<E::Key>: Eq + Hash,
{
    fn as_storage(&self) -> &StorageTarget<E> {
        self
    }
}

impl<E> AsStorage<E> for HashStorage<E, Static>
where
    E: Entity<Storage = Self>,
    InnerKey<E::Key>: Eq + Hash,
{
    fn as_storage(&self) -> &StorageTarget<E> {
        self
    }
}

impl<E> AsStorageMut<E> for HashStorage<E, Dynamic>
where
    E: Entity<Storage = Self>,
    InnerKey<E::Key>: Eq + Hash,
{
    fn as_storage_mut(&mut self) -> &mut StorageTarget<E> {
        self
    }
}

impl<E> AsStorageMut<E> for HashStorage<E, Static>
where
    E: Entity<Storage = Self>,
    InnerKey<E::Key>: Eq + Hash,
{
    fn as_storage_mut(&mut self) -> &mut StorageTarget<E> {
        self
    }
}

impl<E, P> Default for HashStorage<E, P>
where
    E: Entity,
    P: Mode,
{
    fn default() -> Self {
        HashStorage {
            inner: Default::default(),
            phantom: PhantomData,
        }
    }
}

#[cfg(not(all(nightly, feature = "unstable")))]
impl<E> Dispatch<E> for HashStorage<E, Dynamic>
where
    E: Entity<Storage = Self>,
    InnerKey<E::Key>: Eq + Hash,
{
    type Target = dyn 'static + DependentStorage<E>;
}

#[cfg(all(nightly, feature = "unstable"))]
#[rustfmt::skip]
impl<E> Dispatch<E> for HashStorage<E, Dynamic>
where
    E: Entity<Storage = Self>,
    InnerKey<E::Key>: Eq + Hash,
{
    type Target<'a> where E: 'a = dyn 'a + DependentStorage<E>;
}

#[cfg(not(all(nightly, feature = "unstable")))]
impl<E> Dispatch<E> for HashStorage<E, Static>
where
    E: Entity<Storage = Self>,
    InnerKey<E::Key>: Eq + Hash,
{
    type Target = Self;
}

#[cfg(all(nightly, feature = "unstable"))]
#[rustfmt::skip]
impl<E> Dispatch<E> for HashStorage<E, Static>
where
    E: Entity<Storage = Self>,
    InnerKey<E::Key>: Eq + Hash,
{
    type Target<'a> where E: 'a = Self;
}

impl<E, P> Enumerate<E> for HashStorage<E, P>
where
    E: Entity,
    InnerKey<E::Key>: Eq + Hash,
    P: Mode,
{
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn iter<'a>(&'a self) -> Box<dyn 'a + Iterator<Item = (E::Key, &E)>> {
        Box::new(
            self.inner
                .iter()
                .map(|(key, entity)| (E::Key::from_inner(*key), entity)),
        )
    }

    fn iter_mut<'a>(&'a mut self) -> Box<dyn 'a + Iterator<Item = (E::Key, &mut E::Data)>>
    where
        E: Payload,
    {
        Box::new(
            self.inner
                .iter_mut()
                .map(|(key, entity)| (E::Key::from_inner(*key), entity.get_mut())),
        )
    }
}

impl<E, P> Get<E> for HashStorage<E, P>
where
    E: Entity,
    InnerKey<E::Key>: Eq + Hash,
    P: Mode,
{
    fn get(&self, key: &E::Key) -> Option<&E> {
        self.inner.get(&key.into_inner())
    }

    fn get_mut(&mut self, key: &E::Key) -> Option<&mut E> {
        self.inner.get_mut(&key.into_inner())
    }
}

impl<E, P> InsertWithKey<E> for HashStorage<E, P>
where
    E: Entity,
    InnerKey<E::Key>: Eq + Hash,
    P: Mode,
{
    fn insert_with_key(&mut self, key: &E::Key, entity: E) -> Option<E> {
        self.inner.insert(key.into_inner(), entity)
    }
}

impl<E, P> Remove<E> for HashStorage<E, P>
where
    E: Entity,
    InnerKey<E::Key>: Eq + Hash,
    P: Mode,
{
    fn remove(&mut self, key: &E::Key) -> Option<E> {
        self.inner.remove(&key.into_inner())
    }
}
