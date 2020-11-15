use slotmap::hop::HopSlotMap;
use std::marker::PhantomData;

use crate::entity::storage::rekey::{Rekey, Rekeying};
use crate::entity::storage::{
    AsStorage, AsStorageMut, Dispatch, Dynamic, Enumerate, Get, IndependentStorage, InnerKey,
    Insert, Key, Mode, Remove, Static, StorageTarget,
};
use crate::entity::{Entity, Payload};

pub use slotmap::Key as SlotKey;

pub struct SlotStorage<E, P = Static>
where
    E: Entity,
    InnerKey<E::Key>: SlotKey,
    P: Mode,
{
    inner: HopSlotMap<InnerKey<<E as Entity>::Key>, E>,
    phantom: PhantomData<P>,
}

impl<E, P> SlotStorage<E, P>
where
    E: Entity,
    InnerKey<E::Key>: SlotKey,
    P: Mode,
{
    pub fn clone_and_set_keys<R>(&self, rekeying: &mut R) -> Self
    where
        R: Rekeying,
        R::Key: From<E::Key>,
    {
        let mut inner = HopSlotMap::with_capacity_and_key(self.inner.len());
        for (key, entity) in self.inner.iter() {
            rekeying.set(Key::from_inner(key), Key::from_inner(inner.insert(*entity)));
        }
        SlotStorage {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<E> AsStorage<E> for SlotStorage<E, Dynamic>
where
    E: Entity<Storage = Self>,
    InnerKey<E::Key>: SlotKey,
{
    fn as_storage(&self) -> &StorageTarget<E> {
        self
    }
}

impl<E> AsStorage<E> for SlotStorage<E, Static>
where
    E: Entity<Storage = Self>,
    InnerKey<E::Key>: SlotKey,
{
    fn as_storage(&self) -> &StorageTarget<E> {
        self
    }
}

impl<E> AsStorageMut<E> for SlotStorage<E, Dynamic>
where
    E: Entity<Storage = Self>,
    InnerKey<E::Key>: SlotKey,
{
    fn as_storage_mut(&mut self) -> &mut StorageTarget<E> {
        self
    }
}

impl<E> AsStorageMut<E> for SlotStorage<E, Static>
where
    E: Entity<Storage = Self>,
    InnerKey<E::Key>: SlotKey,
{
    fn as_storage_mut(&mut self) -> &mut StorageTarget<E> {
        self
    }
}

impl<E, P> Default for SlotStorage<E, P>
where
    E: Entity,
    InnerKey<E::Key>: SlotKey,
    P: Mode,
{
    fn default() -> Self {
        SlotStorage {
            inner: Default::default(),
            phantom: PhantomData,
        }
    }
}

#[cfg(not(all(nightly, feature = "unstable")))]
impl<E> Dispatch<E> for SlotStorage<E, Dynamic>
where
    E: Entity<Storage = Self>,
    InnerKey<E::Key>: SlotKey,
{
    type Target = dyn 'static + IndependentStorage<E>;
}

#[cfg(all(nightly, feature = "unstable"))]
#[rustfmt::skip]
impl<E> Dispatch<E> for SlotStorage<E, Dynamic>
where
    E: Entity<Storage = Self>,
    InnerKey<E::Key>: SlotKey,
{
    type Target<'a> where E: 'a = dyn 'a + IndependentStorage<E>;
}

#[cfg(not(all(nightly, feature = "unstable")))]
impl<E> Dispatch<E> for SlotStorage<E, Static>
where
    E: Entity<Storage = Self>,
    InnerKey<E::Key>: SlotKey,
{
    type Target = Self;
}

#[cfg(all(nightly, feature = "unstable"))]
#[rustfmt::skip]
impl<E> Dispatch<E> for SlotStorage<E, Static>
where
    E: Entity<Storage = Self>,
    InnerKey<E::Key>: SlotKey,
{
    type Target<'a> where E: 'a = Self;
}

impl<E, P> Enumerate<E> for SlotStorage<E, P>
where
    E: Entity,
    InnerKey<E::Key>: SlotKey,
    P: Mode,
{
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn iter<'a>(&'a self) -> Box<dyn 'a + Iterator<Item = (E::Key, &E)>> {
        Box::new(
            self.inner
                .iter()
                .map(|(key, entity)| (E::Key::from_inner(key), entity)),
        )
    }

    fn iter_mut<'a>(&'a mut self) -> Box<dyn 'a + Iterator<Item = (E::Key, &mut E::Data)>>
    where
        E: Payload,
    {
        Box::new(
            self.inner
                .iter_mut()
                .map(|(key, entity)| (E::Key::from_inner(key), entity.get_mut())),
        )
    }
}

impl<E, P> Get<E> for SlotStorage<E, P>
where
    E: Entity,
    InnerKey<E::Key>: SlotKey,
    P: Mode,
{
    fn get(&self, key: &E::Key) -> Option<&E> {
        self.inner.get(key.into_inner())
    }

    fn get_mut(&mut self, key: &E::Key) -> Option<&mut E> {
        self.inner.get_mut(key.into_inner())
    }
}

impl<E, P> Insert<E> for SlotStorage<E, P>
where
    E: Entity,
    InnerKey<E::Key>: SlotKey,
    P: Mode,
{
    fn insert(&mut self, entity: E) -> E::Key {
        E::Key::from_inner(self.inner.insert(entity))
    }
}

impl<E, P, K> Rekey<K> for SlotStorage<E, P>
where
    E: Entity + Rekey<K>,
    InnerKey<E::Key>: SlotKey,
    P: Mode,
    K: Copy + Eq,
{
    fn rekey(&mut self, rekeying: &impl Rekeying<Key = K>) -> bool {
        let mut is_rekeyed = false;
        for entity in self.inner.values_mut() {
            is_rekeyed = is_rekeyed || entity.rekey(rekeying);
        }
        is_rekeyed
    }
}

impl<E, P> Remove<E> for SlotStorage<E, P>
where
    E: Entity,
    InnerKey<E::Key>: SlotKey,
    P: Mode,
{
    fn remove(&mut self, key: &E::Key) -> Option<E> {
        self.inner.remove(key.into_inner())
    }
}
