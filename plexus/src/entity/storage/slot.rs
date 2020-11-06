use slotmap::hop::HopSlotMap;

use crate::entity::storage::{
    AsStorage, AsStorageMut, Dispatch, Enumerate, Get, IndependentStorage, InnerKey, Insert, Key,
    Remove, StorageTarget,
};
use crate::entity::{Entity, Payload};

pub use slotmap::Key as SlotKey;

pub type SlotEntityMap<E> = HopSlotMap<InnerKey<<E as Entity>::Key>, E>;

impl<E, K> AsStorage<E> for HopSlotMap<InnerKey<K>, E>
where
    E: Entity<Key = K, Storage = Self>,
    K: Key,
    InnerKey<K>: 'static + SlotKey,
{
    fn as_storage(&self) -> &StorageTarget<E> {
        self
    }
}

impl<E, K> AsStorageMut<E> for HopSlotMap<InnerKey<K>, E>
where
    E: Entity<Key = K, Storage = Self>,
    K: Key,
    InnerKey<K>: 'static + SlotKey,
{
    fn as_storage_mut(&mut self) -> &mut StorageTarget<E> {
        self
    }
}

#[cfg(not(all(nightly, feature = "unstable")))]
impl<E, K> Dispatch<E> for HopSlotMap<InnerKey<K>, E>
where
    E: Entity<Key = K, Storage = Self>,
    K: Key,
    InnerKey<K>: 'static + SlotKey,
{
    type Target = dyn 'static + IndependentStorage<E>;
}

#[cfg(all(nightly, feature = "unstable"))]
#[rustfmt::skip]
impl<E, K> Dispatch<E> for HopSlotMap<InnerKey<K>, E>
where
    E: Entity<Key = K, Storage = Self>,
    K: Key,
    InnerKey<K>: 'static + SlotKey,
{
    type Target<'a> where E: 'a = dyn 'a + IndependentStorage<E>;
}

impl<E> Enumerate<E> for HopSlotMap<InnerKey<E::Key>, E>
where
    E: Entity,
    InnerKey<E::Key>: SlotKey,
{
    fn len(&self) -> usize {
        self.len()
    }

    fn iter<'a>(&'a self) -> Box<dyn 'a + Iterator<Item = (E::Key, &E)>> {
        Box::new(
            self.iter()
                .map(|(key, entity)| (E::Key::from_inner(key), entity)),
        )
    }

    fn iter_mut<'a>(&'a mut self) -> Box<dyn 'a + Iterator<Item = (E::Key, &mut E::Data)>>
    where
        E: Payload,
    {
        Box::new(
            self.iter_mut()
                .map(|(key, entity)| (E::Key::from_inner(key), entity.get_mut())),
        )
    }
}

impl<E> Get<E> for HopSlotMap<InnerKey<E::Key>, E>
where
    E: Entity,
    InnerKey<E::Key>: SlotKey,
{
    fn get(&self, key: &E::Key) -> Option<&E> {
        self.get(key.into_inner())
    }

    fn get_mut(&mut self, key: &E::Key) -> Option<&mut E> {
        self.get_mut(key.into_inner())
    }
}

impl<E> Insert<E> for HopSlotMap<InnerKey<E::Key>, E>
where
    E: Entity,
    InnerKey<E::Key>: SlotKey,
{
    fn insert(&mut self, entity: E) -> E::Key {
        E::Key::from_inner(self.insert(entity))
    }
}

impl<E> Remove<E> for HopSlotMap<InnerKey<E::Key>, E>
where
    E: Entity,
    InnerKey<E::Key>: SlotKey,
{
    fn remove(&mut self, key: &E::Key) -> Option<E> {
        self.remove(key.into_inner())
    }
}
