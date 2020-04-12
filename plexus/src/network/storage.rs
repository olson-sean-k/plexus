use fnv::FnvBuildHasher;
use slotmap::hop::HopSlotMap;
use slotmap::Key as SlotKey;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hash};

use crate::network::Entity;

pub type Rekeying<E> = HashMap<<E as Entity>::Key, <E as Entity>::Key, FnvBuildHasher>;
pub type InnerKey<K> = <K as OpaqueKey>::Inner;

pub type SlotStorage<E> = HopSlotMap<InnerKey<<E as Entity>::Key>, E>;
pub type HashStorage<E> = HashMap<InnerKey<<E as Entity>::Key>, E, FnvBuildHasher>;

pub trait OpaqueKey: Copy + Eq + Hash + Sized {
    type Inner: Copy + Sized;

    fn from_inner(key: Self::Inner) -> Self;

    fn into_inner(self) -> Self::Inner;
}

pub trait Fuse<M, T>
where
    M: AsStorage<T>,
    T: Entity,
{
    type Output: AsStorage<T>;

    fn fuse(self, source: M) -> Self::Output;
}

pub trait AsStorage<E>
where
    E: Entity,
{
    fn as_storage(&self) -> &Storage<E>;
}

impl<'a, E, U> AsStorage<E> for &'a U
where
    E: Entity,
    U: AsStorage<E>,
{
    fn as_storage(&self) -> &Storage<E> {
        <U as AsStorage<E>>::as_storage(self)
    }
}

impl<'a, E, U> AsStorage<E> for &'a mut U
where
    E: Entity,
    U: AsStorage<E>,
{
    fn as_storage(&self) -> &Storage<E> {
        <U as AsStorage<E>>::as_storage(self)
    }
}

pub trait AsStorageMut<E>: AsStorage<E>
where
    E: Entity,
{
    fn as_storage_mut(&mut self) -> &mut Storage<E>;
}

impl<'a, E, U> AsStorageMut<E> for &'a mut U
where
    E: Entity,
    U: AsStorageMut<E>,
{
    fn as_storage_mut(&mut self) -> &mut Storage<E> {
        <U as AsStorageMut<E>>::as_storage_mut(self)
    }
}

pub trait AsStorageOf {
    fn as_storage_of<E>(&self) -> &Storage<E>
    where
        E: Entity,
        Self: AsStorage<E>,
    {
        self.as_storage()
    }
}

impl<T> AsStorageOf for T {}

pub trait AsStorageMutOf {
    fn as_storage_mut_of<E>(&mut self) -> &mut Storage<E>
    where
        E: Entity,
        Self: AsStorageMut<E>,
    {
        self.as_storage_mut()
    }
}

impl<T> AsStorageMutOf for T {}

// TODO: Avoid boxing when GATs are stabilized. See
//       https://github.com/rust-lang/rust/issues/44265
pub trait Sequence<E>
where
    E: Entity,
{
    fn len(&self) -> usize;

    fn iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = (E::Key, &E)> + 'a>;

    fn iter_mut<'a>(&'a mut self) -> Box<dyn ExactSizeIterator<Item = (E::Key, &mut E)> + 'a>;

    fn keys<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = E::Key> + 'a>;
}

pub trait Get<E>
where
    E: Entity,
{
    fn get(&self, key: &E::Key) -> Option<&E>;

    fn get_mut(&mut self, key: &E::Key) -> Option<&mut E>;
}

pub trait Remove<E>
where
    E: Entity,
{
    fn remove(&mut self, key: &E::Key) -> Option<E>;
}

pub trait Insert<E>
where
    E: Entity,
{
    fn insert(&mut self, entity: E) -> E::Key;
}

pub trait InsertWithKey<E>
where
    E: Entity,
{
    fn insert_with_key(&mut self, key: E::Key, entity: E) -> Option<E>;
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

impl<E, H> Sequence<E> for HashMap<InnerKey<E::Key>, E, H>
where
    E: Entity,
    H: BuildHasher + Default,
    InnerKey<E::Key>: Eq + Hash,
{
    fn len(&self) -> usize {
        self.len()
    }

    fn iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = (E::Key, &E)> + 'a> {
        Box::new(
            self.iter()
                .map(|(key, entity)| (E::Key::from_inner(*key), entity)),
        )
    }

    fn iter_mut<'a>(&'a mut self) -> Box<dyn ExactSizeIterator<Item = (E::Key, &mut E)> + 'a> {
        Box::new(
            self.iter_mut()
                .map(|(key, entity)| (E::Key::from_inner(*key), entity)),
        )
    }

    fn keys<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = E::Key> + 'a> {
        Box::new(self.keys().map(|key| E::Key::from_inner(*key)))
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

impl<E, H> InsertWithKey<E> for HashMap<InnerKey<E::Key>, E, H>
where
    E: Entity,
    H: BuildHasher + Default,
    InnerKey<E::Key>: Eq + Hash,
{
    fn insert_with_key(&mut self, key: E::Key, entity: E) -> Option<E> {
        self.insert(key.into_inner(), entity)
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

impl<E> Sequence<E> for HopSlotMap<InnerKey<E::Key>, E>
where
    E: Entity,
    InnerKey<E::Key>: SlotKey,
{
    fn len(&self) -> usize {
        self.len()
    }

    fn iter<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = (E::Key, &E)> + 'a> {
        Box::new(
            self.iter()
                .map(|(key, entity)| (E::Key::from_inner(key), entity)),
        )
    }

    fn iter_mut<'a>(&'a mut self) -> Box<dyn ExactSizeIterator<Item = (E::Key, &mut E)> + 'a> {
        Box::new(
            self.iter_mut()
                .map(|(key, entity)| (E::Key::from_inner(key), entity)),
        )
    }

    fn keys<'a>(&'a self) -> Box<dyn ExactSizeIterator<Item = E::Key> + 'a> {
        Box::new(self.keys().map(E::Key::from_inner))
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

impl<E> Insert<E> for HopSlotMap<InnerKey<E::Key>, E>
where
    E: Entity,
    InnerKey<E::Key>: SlotKey,
{
    fn insert(&mut self, entity: E) -> E::Key {
        E::Key::from_inner(self.insert(entity))
    }
}

#[derive(Clone, Default)]
pub struct Storage<E>
where
    E: Entity,
{
    inner: E::Storage,
}

impl<E> Storage<E>
where
    E: Entity,
{
    pub fn new() -> Self {
        Storage {
            inner: Default::default(),
        }
    }

    pub fn from_inner(inner: E::Storage) -> Self {
        Storage { inner }
    }

    pub fn into_inner(self) -> E::Storage {
        let Storage { inner } = self;
        inner
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    // TODO: Use a `Clone` bound.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (E::Key, &E)> {
        self.inner.iter()
    }

    pub fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = (E::Key, &mut E)> {
        self.inner.iter_mut()
    }

    // TODO: Use a `Clone` bound.
    pub fn keys<'a>(&'a self) -> impl ExactSizeIterator<Item = E::Key> + 'a {
        self.inner.keys()
    }

    pub fn contains_key(&self, key: &E::Key) -> bool {
        self.inner.get(key).is_some()
    }

    pub fn get(&self, key: &E::Key) -> Option<&E> {
        self.inner.get(key)
    }

    pub fn get_mut(&mut self, key: &E::Key) -> Option<&mut E> {
        self.inner.get_mut(key)
    }

    pub fn insert(&mut self, entity: E) -> E::Key
    where
        E::Storage: Insert<E>,
    {
        self.inner.insert(entity)
    }

    pub fn insert_with_key(&mut self, key: E::Key, entity: E) -> Option<E>
    where
        E::Storage: InsertWithKey<E>,
    {
        self.inner.insert_with_key(key, entity)
    }

    pub fn remove(&mut self, key: &E::Key) -> Option<E> {
        self.inner.remove(key)
    }

    pub fn partition_with<F>(self, f: F) -> (Self, Self)
    where
        E::Storage: Default + Extend<(E::Key, E)> + IntoIterator<Item = (E::Key, E)>,
        F: FnMut(&(E::Key, E)) -> bool,
    {
        let (left, right) = self.into_inner().into_iter().partition(f);
        (Self::from_inner(left), Self::from_inner(right))
    }

    pub fn partition_rekey_with<F>(self, f: F) -> (Rekeying<E>, (Self, Self))
    where
        E::Key: Eq + Hash,
        E::Storage: Insert<E> + IntoIterator<Item = (E::Key, E)>,
        F: FnMut(&(E::Key, E)) -> bool,
    {
        let mut rekeying = Rekeying::<E>::default();
        let mut rekey = |partition| {
            let mut storage = Self::new();
            for (key, entity) in partition {
                rekeying.insert(key, storage.insert(entity));
            }
            storage
        };
        let (left, right): (Vec<_>, Vec<_>) = self.into_inner().into_iter().partition(f);
        let left = rekey(left);
        let right = rekey(right);
        (rekeying, (left, right))
    }
}

impl<E> AsStorage<E> for Storage<E>
where
    E: Entity,
{
    fn as_storage(&self) -> &Storage<E> {
        self
    }
}

impl<E> AsStorageMut<E> for Storage<E>
where
    E: Entity,
{
    fn as_storage_mut(&mut self) -> &mut Storage<E> {
        self
    }
}
