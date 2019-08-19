//! Storage for topological and payload data in a graph.
//!
//! Graphs abstract their storage behind `StorageProxy` and the `AsStorage`
//! trait. This allows internal APIs to manipulate graphs represented using
//! different types (e.g., `Core`) and to operate on individual storage
//! instances. In particular, the `mutation` module's API takes advantage of
//! this to decompose, mutate, and recompose graphs.
//!
//! `StorageProxy` supports different container types. Slot maps are used for
//! most data, but hash maps are used for arcs. This allows for $O(1)$ queries
//! for arcs, which is a core design assumption of `MeshGraph`.

use fnv::FnvBuildHasher;
use slotmap::hop::HopSlotMap;
use slotmap::Key as SlotKey;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hash};

use crate::graph::storage::key::{InnerKey, OpaqueKey};
use crate::graph::storage::payload::Payload;

pub mod alias;
pub mod key;
pub mod payload;

pub type SlotStorage<T> = HopSlotMap<InnerKey<<T as Payload>::Key>, T>;
pub type HashStorage<T> = HashMap<InnerKey<<T as Payload>::Key>, T, FnvBuildHasher>;

pub type Rekeying<T> = HashMap<<T as Payload>::Key, <T as Payload>::Key, FnvBuildHasher>;

pub trait AsStorage<T>
where
    T: Payload,
{
    fn as_storage(&self) -> &StorageProxy<T>;
}

impl<'a, T, U> AsStorage<T> for &'a U
where
    T: Payload,
    U: AsStorage<T>,
{
    fn as_storage(&self) -> &StorageProxy<T> {
        <U as AsStorage<T>>::as_storage(self)
    }
}

impl<'a, T, U> AsStorage<T> for &'a mut U
where
    T: Payload,
    U: AsStorage<T>,
{
    fn as_storage(&self) -> &StorageProxy<T> {
        <U as AsStorage<T>>::as_storage(self)
    }
}

pub trait AsStorageMut<T>
where
    T: Payload,
{
    fn as_storage_mut(&mut self) -> &mut StorageProxy<T>;
}

impl<'a, T, U> AsStorageMut<T> for &'a mut U
where
    T: Payload,
    U: AsStorageMut<T>,
{
    fn as_storage_mut(&mut self) -> &mut StorageProxy<T> {
        <U as AsStorageMut<T>>::as_storage_mut(self)
    }
}

// TODO: Avoid boxing when GATs are stabilized. See
//       https://github.com/rust-lang/rust/issues/44265
pub trait Sequence<T>
where
    T: Payload,
{
    fn len(&self) -> usize;

    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = (T::Key, &T)> + 'a>;

    fn iter_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = (T::Key, &mut T)> + 'a>;

    fn keys<'a>(&'a self) -> Box<dyn Iterator<Item = T::Key> + 'a>;
}

pub trait Get<T>
where
    T: Payload,
{
    fn get(&self, key: &T::Key) -> Option<&T>;

    fn get_mut(&mut self, key: &T::Key) -> Option<&mut T>;
}

pub trait Remove<T>
where
    T: Payload,
{
    fn remove(&mut self, key: &T::Key) -> Option<T>;
}

pub trait Insert<T>
where
    T: Payload,
{
    fn insert(&mut self, payload: T) -> T::Key;
}

pub trait InsertWithKey<T>
where
    T: Payload,
{
    fn insert_with_key(&mut self, key: T::Key, payload: T) -> Option<T>;
}

impl<T, H> Get<T> for HashMap<InnerKey<T::Key>, T, H>
where
    T: Payload,
    H: BuildHasher + Default,
    InnerKey<T::Key>: Eq + Hash,
{
    fn get(&self, key: &T::Key) -> Option<&T> {
        self.get(&key.into_inner())
    }

    fn get_mut(&mut self, key: &T::Key) -> Option<&mut T> {
        self.get_mut(&key.into_inner())
    }
}

impl<T, H> Sequence<T> for HashMap<InnerKey<T::Key>, T, H>
where
    T: Payload,
    H: BuildHasher + Default,
    InnerKey<T::Key>: Eq + Hash,
{
    fn len(&self) -> usize {
        self.len()
    }

    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = (T::Key, &T)> + 'a> {
        Box::new(
            self.iter()
                .map(|(key, payload)| (T::Key::from_inner(*key), payload)),
        )
    }

    fn iter_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = (T::Key, &mut T)> + 'a> {
        Box::new(
            self.iter_mut()
                .map(|(key, payload)| (T::Key::from_inner(*key), payload)),
        )
    }

    fn keys<'a>(&'a self) -> Box<dyn Iterator<Item = T::Key> + 'a> {
        Box::new(self.keys().map(|key| T::Key::from_inner(*key)))
    }
}

impl<T, H> Remove<T> for HashMap<InnerKey<T::Key>, T, H>
where
    T: Payload,
    H: BuildHasher + Default,
    InnerKey<T::Key>: Eq + Hash,
{
    fn remove(&mut self, key: &T::Key) -> Option<T> {
        self.remove(&key.into_inner())
    }
}

impl<T, H> InsertWithKey<T> for HashMap<InnerKey<T::Key>, T, H>
where
    T: Payload,
    H: BuildHasher + Default,
    InnerKey<T::Key>: Eq + Hash,
{
    fn insert_with_key(&mut self, key: T::Key, payload: T) -> Option<T> {
        self.insert(key.into_inner(), payload)
    }
}

impl<T> Get<T> for HopSlotMap<InnerKey<T::Key>, T>
where
    T: Payload,
    InnerKey<T::Key>: SlotKey,
{
    fn get(&self, key: &T::Key) -> Option<&T> {
        self.get(key.into_inner())
    }

    fn get_mut(&mut self, key: &T::Key) -> Option<&mut T> {
        self.get_mut(key.into_inner())
    }
}

impl<T> Sequence<T> for HopSlotMap<InnerKey<T::Key>, T>
where
    T: Payload,
    InnerKey<T::Key>: SlotKey,
{
    fn len(&self) -> usize {
        self.len()
    }

    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = (T::Key, &T)> + 'a> {
        Box::new(
            self.iter()
                .map(|(key, payload)| (T::Key::from_inner(key), payload)),
        )
    }

    fn iter_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = (T::Key, &mut T)> + 'a> {
        Box::new(
            self.iter_mut()
                .map(|(key, payload)| (T::Key::from_inner(key), payload)),
        )
    }

    fn keys<'a>(&'a self) -> Box<dyn Iterator<Item = T::Key> + 'a> {
        Box::new(self.keys().map(T::Key::from_inner))
    }
}

impl<T> Remove<T> for HopSlotMap<InnerKey<T::Key>, T>
where
    T: Payload,
    InnerKey<T::Key>: SlotKey,
{
    fn remove(&mut self, key: &T::Key) -> Option<T> {
        self.remove(key.into_inner())
    }
}

impl<T> Insert<T> for HopSlotMap<InnerKey<T::Key>, T>
where
    T: Payload,
    InnerKey<T::Key>: SlotKey,
{
    fn insert(&mut self, payload: T) -> T::Key {
        T::Key::from_inner(self.insert(payload))
    }
}

// TODO: The `FromInteriorGeometry` trait is far less useful without being able
//       to map over storage. Implement mapping or consider removing
//       `FromInteriorGeometry` and its related traits.
#[derive(Clone, Default)]
pub struct StorageProxy<T>
where
    T: Payload,
{
    inner: T::Storage,
}

impl<T> StorageProxy<T>
where
    T: Payload,
{
    pub fn new() -> Self {
        StorageProxy {
            inner: Default::default(),
        }
    }

    pub fn from_inner(inner: T::Storage) -> Self {
        StorageProxy { inner }
    }

    pub fn into_inner(self) -> T::Storage {
        self.inner
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    // TODO: Return `Clone + Iterator`.
    pub fn iter(&self) -> impl Iterator<Item = (T::Key, &T)> {
        self.inner.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (T::Key, &mut T)> {
        self.inner.iter_mut()
    }

    // TODO: Return `Clone + Iterator`.
    pub fn keys<'a>(&'a self) -> impl Iterator<Item = T::Key> + 'a {
        self.inner.keys()
    }

    pub fn contains_key(&self, key: &T::Key) -> bool {
        self.inner.get(key).is_some()
    }

    pub fn get(&self, key: &T::Key) -> Option<&T> {
        self.inner.get(key)
    }

    pub fn get_mut(&mut self, key: &T::Key) -> Option<&mut T> {
        self.inner.get_mut(key)
    }

    pub fn insert(&mut self, payload: T) -> T::Key
    where
        T::Storage: Insert<T>,
    {
        self.inner.insert(payload)
    }

    pub fn insert_with_key(&mut self, key: T::Key, payload: T) -> Option<T>
    where
        T::Storage: InsertWithKey<T>,
    {
        self.inner.insert_with_key(key, payload)
    }

    pub fn remove(&mut self, key: &T::Key) -> Option<T> {
        self.inner.remove(key)
    }

    pub fn partition_with<F>(self, f: F) -> (Self, Self)
    where
        T::Storage: Default + Extend<(T::Key, T)> + IntoIterator<Item = (T::Key, T)>,
        F: FnMut(&(T::Key, T)) -> bool,
    {
        let (left, right) = self.into_inner().into_iter().partition(f);
        (Self::from_inner(left), Self::from_inner(right))
    }

    pub fn partition_rekey_with<F>(self, f: F) -> (Rekeying<T>, (Self, Self))
    where
        T::Key: Eq + Hash,
        T::Storage: Insert<T> + IntoIterator<Item = (T::Key, T)>,
        F: FnMut(&(T::Key, T)) -> bool,
    {
        let mut rekeying = Rekeying::<T>::default();
        let mut rekey = |partition| {
            let mut storage = Self::new();
            for (key, payload) in partition {
                rekeying.insert(key, storage.insert(payload));
            }
            storage
        };
        let (left, right): (Vec<_>, Vec<_>) = self.into_inner().into_iter().partition(f);
        let left = rekey(left);
        let right = rekey(right);
        (rekeying, (left, right))
    }
}

impl<T> AsStorage<T> for StorageProxy<T>
where
    T: Payload,
{
    fn as_storage(&self) -> &StorageProxy<T> {
        self
    }
}

impl<T> AsStorageMut<T> for StorageProxy<T>
where
    T: Payload,
{
    fn as_storage_mut(&mut self) -> &mut StorageProxy<T> {
        self
    }
}
