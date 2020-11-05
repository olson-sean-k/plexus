mod hash;
mod journal;
mod slot;

use std::hash::Hash;

use crate::entity::{Entity, Payload};

pub use crate::entity::storage::hash::FnvEntityMap;
pub use crate::entity::storage::journal::{Journaled, Rekeying, Unjournaled};
pub use crate::entity::storage::slot::SlotEntityMap;

#[cfg(not(all(nightly, feature = "unstable")))]
pub type StorageTarget<E> = <<E as Entity>::Storage as Dispatch<E>>::Target;
#[cfg(all(nightly, feature = "unstable"))]
pub type StorageTarget<'a, E> = <<E as Entity>::Storage as Dispatch<E>>::Target<'a>;

pub type InnerKey<K> = <K as Key>::Inner;

pub trait Key: Copy + Eq + Hash + Sized {
    type Inner: Copy + Sized;

    fn from_inner(key: Self::Inner) -> Self;

    fn into_inner(self) -> Self::Inner;
}

pub trait DependentKey: Key {
    type Foreign: Key;

    fn rekey(self, rekeying: &Rekeying<Self::Foreign>) -> Self;
}

#[cfg(not(all(nightly, feature = "unstable")))]
pub trait Dispatch<E>
where
    E: Entity,
{
    type Target: ?Sized + Storage<E>;
}

#[cfg(all(nightly, feature = "unstable"))]
#[rustfmt::skip]
pub trait Dispatch<E>
where
    E: Entity,
{
    type Target<'a>: 'a + ?Sized + Storage<E>
    where
        E: 'a;
}

// TODO: Can GATs be used here while still supporting trait objects?
pub trait Enumerate<E>
where
    E: Entity,
{
    fn len(&self) -> usize;

    fn iter<'a>(&'a self) -> Box<dyn 'a + Iterator<Item = (E::Key, &E)>>;

    // This iterator exposes mutable references to user data and **not**
    // entities. This prevents categorical mutations of entities, which
    // interacts poorly with journaling. Moreover, such an iterator does not
    // provide much utility, because entities are typically mutated via
    // relationships like adjacency.
    fn iter_mut<'a>(&'a mut self) -> Box<dyn 'a + Iterator<Item = (E::Key, &mut E::Data)>>
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

impl<'a, E, T> AsStorage<E> for &'a T
where
    E: Entity,
    T: AsStorage<E> + ?Sized,
{
    fn as_storage(&self) -> &StorageTarget<E> {
        <T as AsStorage<E>>::as_storage(self)
    }
}

impl<'a, E, T> AsStorage<E> for &'a mut T
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

impl<'a, E, T> AsStorageMut<E> for &'a mut T
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

#[cfg(test)]
mod tests {
    use slotmap::DefaultKey;

    use crate::entity::storage::{DependentKey, FnvEntityMap, Key, Rekeying, SlotEntityMap};
    use crate::entity::Entity;

    #[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
    pub struct NodeKey(DefaultKey);

    impl Key for NodeKey {
        type Inner = DefaultKey;

        fn from_inner(key: Self::Inner) -> Self {
            NodeKey(key)
        }

        fn into_inner(self) -> Self::Inner {
            self.0
        }
    }

    #[derive(Clone, Copy, Default)]
    pub struct Node;

    impl Entity for Node {
        type Key = NodeKey;
        type Storage = SlotEntityMap<Self>;
    }

    #[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
    pub struct LinkKey(NodeKey, NodeKey);

    impl DependentKey for LinkKey {
        type Foreign = NodeKey;

        fn rekey(self, rekeying: &Rekeying<Self::Foreign>) -> Self {
            let LinkKey(a, b) = self;
            let a = rekeying.get(&a).cloned().unwrap_or(a);
            let b = rekeying.get(&b).cloned().unwrap_or(b);
            LinkKey(a, b)
        }
    }

    impl Key for LinkKey {
        type Inner = (NodeKey, NodeKey);

        fn from_inner(key: Self::Inner) -> Self {
            LinkKey(key.0, key.1)
        }

        fn into_inner(self) -> Self::Inner {
            (self.0, self.1)
        }
    }

    #[derive(Clone, Copy, Default)]
    pub struct Link;

    impl Entity for Link {
        type Key = LinkKey;
        type Storage = FnvEntityMap<Self>;
    }
}
