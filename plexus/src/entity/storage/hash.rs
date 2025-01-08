use ahash::AHashMap;
use std::hash::Hash;
use std::marker::PhantomData;

use crate::entity::storage::{
    AsStorage, AsStorageMut, DependentStorage, Dispatch, Dynamic, Enumerate, Get, IncrementalKeyer,
    IndependentStorage, InnerKey, Insert, InsertWithKey, Key, Keyer, Mode, Remove, Static,
    StorageTarget,
};
use crate::entity::{Entity, Payload};

// TODO: The `Keyer` parameter `R` of `HashStorage` cannot be parameterized when
//       implementing the `AsStorage` and `Dispatch` traits even if the
//       conflicting implementations use a private local type or the `Keyer`
//       trait is private. Instead, this is implemented more specifically for
//       `IncrementalKeyer`. Perhaps this will be possible in the future.
//
//       See https://github.com/rust-lang/rust/issues/48869

pub struct HashStorage<E, R = (), P = Static>
where
    E: Entity,
    R: Default,
    P: Mode,
{
    inner: AHashMap<InnerKey<<E as Entity>::Key>, E>,
    keyer: R,
    phantom: PhantomData<fn() -> P>,
}

impl<E, R, P> HashStorage<E, R, P>
where
    E: Entity,
    InnerKey<E::Key>: Eq + Hash,
    R: Default,
    P: Mode,
{
    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit();
    }
}

impl<E> AsStorage<E> for HashStorage<E, (), Dynamic>
where
    E: Entity<Storage = Self>,
    InnerKey<E::Key>: Eq + Hash,
{
    fn as_storage(&self) -> &StorageTarget<E> {
        self
    }
}

impl<E> AsStorage<E> for HashStorage<E, IncrementalKeyer, Dynamic>
where
    E: Entity<Storage = Self>,
    E::Key: Key<Inner = u64>,
{
    fn as_storage(&self) -> &StorageTarget<E> {
        self
    }
}

impl<E, R> AsStorage<E> for HashStorage<E, R, Static>
where
    E: Entity<Storage = Self>,
    InnerKey<E::Key>: Eq + Hash,
    R: 'static + Default,
{
    fn as_storage(&self) -> &StorageTarget<E> {
        self
    }
}

impl<E> AsStorageMut<E> for HashStorage<E, (), Dynamic>
where
    E: Entity<Storage = Self>,
    InnerKey<E::Key>: Eq + Hash,
{
    fn as_storage_mut(&mut self) -> &mut StorageTarget<E> {
        self
    }
}

impl<E> AsStorageMut<E> for HashStorage<E, IncrementalKeyer, Dynamic>
where
    E: Entity<Storage = Self>,
    E::Key: Key<Inner = u64>,
{
    fn as_storage_mut(&mut self) -> &mut StorageTarget<E> {
        self
    }
}

impl<E, R> AsStorageMut<E> for HashStorage<E, R, Static>
where
    E: Entity<Storage = Self>,
    InnerKey<E::Key>: Eq + Hash,
    R: 'static + Default,
{
    fn as_storage_mut(&mut self) -> &mut StorageTarget<E> {
        self
    }
}

impl<E, R, P> Default for HashStorage<E, R, P>
where
    E: Entity,
    R: Default,
    P: Mode,
{
    fn default() -> Self {
        HashStorage {
            inner: Default::default(),
            keyer: Default::default(),
            phantom: PhantomData,
        }
    }
}

#[rustfmt::skip]
impl<E> Dispatch<E> for HashStorage<E, (), Dynamic>
where
    E: Entity<Storage = Self>,
    InnerKey<E::Key>: Eq + Hash,
{
    type Target<'a> = dyn 'a + DependentStorage<E> where E: 'a;
}

#[rustfmt::skip]
impl<E> Dispatch<E> for HashStorage<E, IncrementalKeyer, Dynamic>
where
    E: Entity<Storage = Self>,
    E::Key: Key<Inner = u64>,
{
    type Target<'a> = dyn 'a + IndependentStorage<E> where E: 'a;
}

#[rustfmt::skip]
impl<E, R> Dispatch<E> for HashStorage<E, R, Static>
where
    E: Entity<Storage = Self>,
    InnerKey<E::Key>: Eq + Hash,
    R: 'static + Default,
{
    type Target<'a> = Self where E: 'a;
}

impl<E, R, P> Enumerate<E> for HashStorage<E, R, P>
where
    E: Entity,
    InnerKey<E::Key>: Eq + Hash,
    R: Default,
    P: Mode,
{
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn iter(&self) -> Box<dyn '_ + Iterator<Item = (E::Key, &'_ E)>> {
        Box::new(
            self.inner
                .iter()
                .map(|(key, entity)| (E::Key::from_inner(*key), entity)),
        )
    }

    fn iter_mut(&mut self) -> Box<dyn '_ + Iterator<Item = (E::Key, &'_ mut E::Data)>>
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

impl<E, R, P> Get<E> for HashStorage<E, R, P>
where
    E: Entity,
    InnerKey<E::Key>: Eq + Hash,
    R: Default,
    P: Mode,
{
    fn get(&self, key: &E::Key) -> Option<&E> {
        self.inner.get(&key.into_inner())
    }

    fn get_mut(&mut self, key: &E::Key) -> Option<&mut E> {
        self.inner.get_mut(&key.into_inner())
    }
}

impl<E, R, P> Insert<E> for HashStorage<E, R, P>
where
    E: Entity,
    InnerKey<E::Key>: Eq + Hash,
    R: Keyer<E::Key>,
    P: Mode,
{
    fn insert(&mut self, entity: E) -> E::Key {
        let key = self.keyer.next();
        self.inner.insert(key, entity);
        Key::from_inner(key)
    }
}

impl<E, P> InsertWithKey<E> for HashStorage<E, (), P>
where
    E: Entity,
    InnerKey<E::Key>: Eq + Hash,
    P: Mode,
{
    fn insert_with_key(&mut self, key: &E::Key, entity: E) -> Option<E> {
        self.inner.insert(key.into_inner(), entity)
    }
}

impl<E, R, P> Remove<E> for HashStorage<E, R, P>
where
    E: Entity,
    InnerKey<E::Key>: Eq + Hash,
    R: Default,
    P: Mode,
{
    fn remove(&mut self, key: &E::Key) -> Option<E> {
        self.inner.remove(&key.into_inner())
    }
}
