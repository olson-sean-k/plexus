use fnv::FnvBuildHasher;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hash};

use crate::entity::storage::{
    AsStorage, AsStorageMut, DependentStorage, Dispatch, Enumerate, Get, InnerKey, InsertWithKey,
    Key, Remove, StorageTarget,
};
use crate::entity::{Entity, Payload};

pub type FnvEntityMap<E> = HashMap<InnerKey<<E as Entity>::Key>, E, FnvBuildHasher>;

impl<E, K, H> AsStorage<E> for HashMap<InnerKey<K>, E, H>
where
    E: Entity<Key = K, Storage = Self>,
    K: Key,
    H: 'static + BuildHasher + Default,
    InnerKey<K>: 'static + Eq + Hash,
{
    fn as_storage(&self) -> &StorageTarget<E> {
        self
    }
}

impl<E, K, H> AsStorageMut<E> for HashMap<InnerKey<K>, E, H>
where
    E: Entity<Key = K, Storage = Self>,
    K: Key,
    H: 'static + BuildHasher + Default,
    InnerKey<K>: 'static + Eq + Hash,
{
    fn as_storage_mut(&mut self) -> &mut StorageTarget<E> {
        self
    }
}

#[cfg(not(all(nightly, feature = "unstable")))]
impl<E, K, H> Dispatch<E> for HashMap<InnerKey<K>, E, H>
where
    E: Entity<Key = K, Storage = Self>,
    K: Key,
    H: 'static + BuildHasher + Default,
    InnerKey<K>: 'static + Eq + Hash,
{
    type Target = dyn 'static + DependentStorage<E>;
}

#[cfg(all(nightly, feature = "unstable"))]
#[rustfmt::skip]
impl<E, K, H> Dispatch<E> for HashMap<InnerKey<K>, E, H>
where
    E: Entity<Key = K, Storage = Self>,
    K: Key,
    H: 'static + BuildHasher + Default,
    InnerKey<K>: 'static + Eq + Hash,
{
    type Target<'a> where E: 'a = dyn 'a + DependentStorage<E>;
}

impl<E, H> Enumerate<E> for HashMap<InnerKey<E::Key>, E, H>
where
    E: Entity,
    H: BuildHasher + Default,
    InnerKey<E::Key>: Eq + Hash,
{
    fn len(&self) -> usize {
        self.len()
    }

    fn iter<'a>(&'a self) -> Box<dyn 'a + Iterator<Item = (E::Key, &E)>> {
        Box::new(
            self.iter()
                .map(|(key, entity)| (E::Key::from_inner(*key), entity)),
        )
    }

    fn iter_mut<'a>(&'a mut self) -> Box<dyn 'a + Iterator<Item = (E::Key, &mut E::Data)>>
    where
        E: Payload,
    {
        Box::new(
            self.iter_mut()
                .map(|(key, entity)| (E::Key::from_inner(*key), entity.get_mut())),
        )
    }
}

impl<E, H> Get<E> for HashMap<InnerKey<E::Key>, E, H>
where
    E: Entity,
    H: BuildHasher + Default,
    InnerKey<E::Key>: Eq + Hash,
{
    fn get(&self, key: &E::Key) -> Option<&E> {
        self.get(&key.into_inner())
    }

    fn get_mut(&mut self, key: &E::Key) -> Option<&mut E> {
        self.get_mut(&key.into_inner())
    }
}

impl<E, H> InsertWithKey<E> for HashMap<InnerKey<E::Key>, E, H>
where
    E: Entity,
    H: BuildHasher + Default,
    InnerKey<E::Key>: Eq + Hash,
{
    fn insert_with_key(&mut self, key: &E::Key, entity: E) -> Option<E> {
        self.insert(key.into_inner(), entity)
    }
}

impl<E, H> Remove<E> for HashMap<InnerKey<E::Key>, E, H>
where
    E: Entity,
    H: BuildHasher + Default,
    InnerKey<E::Key>: Eq + Hash,
{
    fn remove(&mut self, key: &E::Key) -> Option<E> {
        self.remove(&key.into_inner())
    }
}
