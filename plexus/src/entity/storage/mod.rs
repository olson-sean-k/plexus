mod hash;

use std::hash::Hash;

use crate::entity::{Entity, Payload};

pub use crate::entity::storage::hash::HashStorage;

pub mod prelude {
    pub use crate::entity::storage::{Enumerate, Get, Insert, InsertWithKey, Remove};
}

pub type StorageTarget<'a, E> = <<E as Entity>::Storage as Dispatch<E>>::Target<'a>;

pub type InnerKey<K> = <K as Key>::Inner;

pub trait Key: Copy + Eq + Hash + Sized {
    type Inner: Copy + Sized;

    fn from_inner(key: Self::Inner) -> Self;

    fn into_inner(self) -> Self::Inner;
}

pub trait Keyer<K>: Clone + Default
where
    K: Key,
{
    fn next(&mut self) -> K::Inner;
}

#[derive(Clone, Copy, Default)]
pub struct IncrementalKeyer {
    key: u64,
}

impl<K> Keyer<K> for IncrementalKeyer
where
    K: Key<Inner = u64>,
{
    fn next(&mut self) -> K::Inner {
        let key = self.key;
        self.key = self.key.checked_add(1).expect("keyspace exhausted");
        key
    }
}

#[rustfmt::skip]
pub trait Dispatch<E>
where
    E: Entity,
{
    type Target<'a>: 'a + ?Sized + Storage<E>
    where
        E: 'a;
}

pub trait Mode {}

pub enum Static {}

impl Mode for Static {}

pub enum Dynamic {}

impl Mode for Dynamic {}

// TODO: GATs cannot yet be used here while still supporting trait objects,
//       because the types cannot be bound in a supertrait. To support trait
//       objects, the associated types must be bound (likely to `Box<dyn ...>`).
//       See https://github.com/rust-lang/rust/issues/67510
pub trait Enumerate<E>
where
    E: Entity,
{
    fn len(&self) -> usize;

    fn iter(&self) -> Box<dyn '_ + Iterator<Item = (E::Key, &'_ E)>>;

    fn iter_mut(&mut self) -> Box<dyn '_ + Iterator<Item = (E::Key, &'_ mut E::Data)>>
    where
        E: Payload;
}

pub trait Get<E>
where
    E: Entity,
{
    fn get(&self, key: &E::Key) -> Option<&E>;

    fn get_mut(&mut self, key: &E::Key) -> Option<&mut E>;

    fn contains_key(&self, key: &E::Key) -> bool {
        self.get(key).is_some()
    }
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
    fn insert_with_key(&mut self, key: &E::Key, entity: E) -> Option<E>;
}

pub trait Remove<E>
where
    E: Entity,
{
    fn remove(&mut self, key: &E::Key) -> Option<E>;
}

pub trait Storage<E>: AsStorage<E> + AsStorageMut<E> + Enumerate<E> + Get<E> + Remove<E>
where
    E: Entity,
{
}

impl<T, E> Storage<E> for T
where
    T: AsStorage<E> + AsStorageMut<E> + Enumerate<E> + Get<E> + Remove<E>,
    E: Entity,
{
}

pub trait DependentStorage<E>: InsertWithKey<E> + Storage<E>
where
    E: Entity,
{
}

impl<T, E> DependentStorage<E> for T
where
    T: InsertWithKey<E> + Storage<E>,
    E: Entity,
{
}

pub trait IndependentStorage<E>: Insert<E> + Storage<E>
where
    E: Entity,
{
}

impl<T, E> IndependentStorage<E> for T
where
    T: Insert<E> + Storage<E>,
    E: Entity,
{
}

pub trait Fuse<M, T>
where
    M: AsStorage<T>,
    T: Entity,
{
    type Output: AsStorage<T>;

    fn fuse(self, source: M) -> Self::Output;
}

// TODO: It may not be possible for storage wrapper types to implement this
//       generally for all underlying storage types when using trait objects. An
//       input type parameter for the wrapped storage type may be required with
//       an implementation for each type that can be wrapped. Is there some way
//       to support blanket implementations?
pub trait AsStorage<E>
where
    E: Entity,
{
    fn as_storage(&self) -> &StorageTarget<E>;
}

impl<E, T> AsStorage<E> for &'_ T
where
    E: Entity,
    T: AsStorage<E> + ?Sized,
{
    fn as_storage(&self) -> &StorageTarget<E> {
        <T as AsStorage<E>>::as_storage(self)
    }
}

impl<E, T> AsStorage<E> for &'_ mut T
where
    E: Entity,
    T: AsStorage<E> + ?Sized,
{
    fn as_storage(&self) -> &StorageTarget<E> {
        <T as AsStorage<E>>::as_storage(self)
    }
}

pub trait AsStorageMut<E>: AsStorage<E>
where
    E: Entity,
{
    fn as_storage_mut(&mut self) -> &mut StorageTarget<E>;
}

impl<E, T> AsStorageMut<E> for &'_ mut T
where
    E: Entity,
    T: AsStorageMut<E> + ?Sized,
{
    fn as_storage_mut(&mut self) -> &mut StorageTarget<E> {
        <T as AsStorageMut<E>>::as_storage_mut(self)
    }
}

pub trait AsStorageOf {
    fn as_storage_of<E>(&self) -> &StorageTarget<E>
    where
        E: Entity,
        Self: AsStorage<E>,
    {
        self.as_storage()
    }

    fn as_storage_mut_of<E>(&mut self) -> &mut StorageTarget<E>
    where
        E: Entity,
        Self: AsStorageMut<E>,
    {
        self.as_storage_mut()
    }
}

impl<T> AsStorageOf for T {}
