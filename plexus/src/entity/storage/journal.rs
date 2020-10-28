use fnv::FnvBuildHasher;
use fool::BoolExt as _;
use ordered_multimap::ListOrderedMultimap as LinkedMultiMap;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use crate::entity::storage::hash::FnvEntityMap;
use crate::entity::storage::slot::{SlotEntityMap, SlotKey};
use crate::entity::storage::{
    AsStorage, AsStorageMut, DependentKey, Dispatch, Enumerate, Get, Insert, InsertWithKey, Key,
    Remove, Storage, StorageObject,
};
use crate::entity::{Entity, Payload};

pub type Rekeying<K> = HashMap<K, K, FnvBuildHasher>;

pub trait Unjournaled {}

pub trait JournalState {
    type State;

    fn state(&self) -> Self::State;
}

pub trait SyntheticKey<T> {
    #[must_use]
    fn synthesize(state: &mut T) -> Self;
}

enum Mutation<E>
where
    E: Entity,
{
    Insert(E),
    Remove,
    Write(E),
}

impl<E> Mutation<E>
where
    E: Entity,
{
    pub fn as_entity(&self) -> Option<&E> {
        match *self {
            Mutation::Insert(ref entity) | Mutation::Write(ref entity) => Some(entity),
            Mutation::Remove => None,
        }
    }

    pub fn as_entity_mut(&mut self) -> Option<&mut E> {
        match *self {
            Mutation::Insert(ref mut entity) | Mutation::Write(ref mut entity) => Some(entity),
            Mutation::Remove => None,
        }
    }
}

// TODO: Should mutations be aggregated in the log? For a given key, the
//       complete history may not be necessary. A `HashMap` or similar data
//       structure may be sufficient for representing the log.
// TODO: The type parameter `T` is only used to implement `AsStorage`. Is there
//       a way to write a generic implementation that also allows for an
//       implicit conversion from `&Journaled<_, _>` to a storage object?
pub struct Journaled<T, E>
where
    T: Default + Dispatch<E> + JournalState + Storage<E> + Unjournaled,
    E: Entity<Storage = T>,
{
    storage: T,
    log: LinkedMultiMap<E::Key, Mutation<E>>,
    state: T::State,
}

impl<T, E> Journaled<T, E>
where
    T: Default + Dispatch<E> + JournalState + Storage<E> + Unjournaled,
    E: Entity<Storage = T>,
{
    pub fn transact(storage: T) -> Self {
        let state = storage.state();
        Journaled {
            storage,
            log: Default::default(),
            state,
        }
    }

    pub fn abort(self) -> T {
        self.storage
    }
}

impl<T, E> Journaled<T, E>
where
    T: Default + Dispatch<E> + Insert<E> + JournalState + Storage<E> + Unjournaled,
    E: Entity<Storage = T>,
    E::Key: SyntheticKey<T::State>,
{
    pub fn commit_and_rekey(self) -> (T, Rekeying<E::Key>) {
        let Journaled {
            mut storage,
            mut log,
            ..
        } = self;
        let rekeying: Rekeying<_> = log
            .drain_pairs()
            .flat_map(|(key, mut entry)| entry.next_back().map(|mutation| (key, mutation)))
            .map(|(key, mutation)| {
                // TODO: Should unmapped keys be inserted into the rekeying? Note
                //       that removing such keys may complicate rekeying of
                //       dependent keys.
                let rekey = match mutation {
                    Mutation::Insert(entity) | Mutation::Write(entity) => {
                        if let Some(occupant) = storage.get_mut(&key) {
                            *occupant = entity;
                            key
                        }
                        else {
                            storage.insert(entity)
                        }
                    }
                    Mutation::Remove => {
                        storage.remove(&key);
                        key
                    }
                };
                (key, rekey)
            })
            .collect();
        (storage, rekeying)
    }
}

impl<T, E> Journaled<T, E>
where
    T: Default + Dispatch<E> + InsertWithKey<E> + JournalState + Storage<E> + Unjournaled,
    E: Entity<Storage = T>,
    E::Key: DependentKey,
{
    pub fn commit_with_rekeying(
        self,
        rekeying: &Rekeying<<E::Key as DependentKey>::Foreign>,
    ) -> (T, Rekeying<E::Key>) {
        let Journaled {
            mut storage,
            mut log,
            ..
        } = self;
        let rekeying: Rekeying<_> = log
            .drain_pairs()
            .flat_map(|(key, mut entry)| entry.next_back().map(|mutation| (key, mutation)))
            .map(|(key, mutation)| {
                let rekey = key.rekey(rekeying);
                match mutation {
                    Mutation::Insert(entity) | Mutation::Write(entity) => {
                        if let Some(occupant) = storage.get_mut(&rekey) {
                            *occupant = entity;
                        }
                        else {
                            storage.insert_with_key(&rekey, entity);
                        }
                    }
                    Mutation::Remove => {
                        storage.remove(&rekey);
                    }
                }
                (key, rekey)
            })
            .collect();
        (storage, rekeying)
    }
}

// TODO: Is a general implementation possible? See `AsStorage`.
impl<E, K> AsStorage<E> for Journaled<FnvEntityMap<E>, E>
where
    E: Entity<Key = K, Storage = FnvEntityMap<E>>,
    K: Key,
    K::Inner: 'static + Eq + Hash,
{
    fn as_storage(&self) -> &StorageObject<E> {
        // It is essential that this returns `self` and does NOT simply forward
        // to the `storage` field.
        self
    }
}

// TODO: Is a general implementation possible? See `AsStorage`.
impl<E, K> AsStorage<E> for Journaled<SlotEntityMap<E>, E>
where
    E: Entity<Key = K, Storage = SlotEntityMap<E>>,
    K: Key,
    K::Inner: 'static + SlotKey,
{
    fn as_storage(&self) -> &StorageObject<E> {
        // It is essential that this returns `self` and does NOT simply forward
        // to the `storage` field.
        self
    }
}

// TODO: Is a general implementation possible? See `AsStorage`.
impl<E, K> AsStorageMut<E> for Journaled<FnvEntityMap<E>, E>
where
    E: Entity<Key = K, Storage = FnvEntityMap<E>>,
    K: Key,
    K::Inner: 'static + Eq + Hash,
{
    fn as_storage_mut(&mut self) -> &mut StorageObject<E> {
        // It is essential that this returns `self` and does NOT simply forward
        // to the `storage` field.
        self
    }
}

// TODO: Is a general implementation possible? See `AsStorage`.
impl<E, K> AsStorageMut<E> for Journaled<SlotEntityMap<E>, E>
where
    E: Entity<Key = K, Storage = SlotEntityMap<E>>,
    K: Key,
    K::Inner: 'static + SlotKey,
{
    fn as_storage_mut(&mut self) -> &mut StorageObject<E> {
        // It is essential that this returns `self` and does NOT simply forward
        // to the `storage` field.
        self
    }
}

#[cfg(not(all(nightly, feature = "unstable")))]
impl<T, E> Dispatch<E> for Journaled<T, E>
where
    T: Default + Dispatch<E> + JournalState + Storage<E> + Unjournaled,
    E: Entity<Storage = T>,
{
    type Object = StorageObject<E>;
}

#[cfg(all(nightly, feature = "unstable"))]
#[rustfmt::skip]
impl<T, E> Dispatch<E> for Journaled<T, E>
where
    T: Default + Dispatch<E> + JournalState + Storage<E> + Unjournaled,
    E: Entity<Storage = T>,
{
    type Object<'a> where E: 'a = StorageObject<'a, E>;
}

impl<T, E> Enumerate<E> for Journaled<T, E>
where
    T: Default + Dispatch<E> + JournalState + Storage<E> + Unjournaled,
    E: Entity<Storage = T>,
{
    fn len(&self) -> usize {
        let n = self.storage.len();
        // Count inserted entities in the log.
        let p = self
            .log
            .pairs()
            .filter_map(|(key, entry)| {
                entry
                    .into_iter()
                    .rev()
                    .find(|mutation| !matches!(mutation, Mutation::Write(_)))
                    .filter(|mutation| matches!(mutation, Mutation::Insert(_)))
                    .filter(|_| !self.storage.contains_key(key))
            })
            .count();
        // Count removed entities in the log.
        let q = self
            .log
            .pairs()
            .filter_map(|(key, entry)| {
                entry
                    .into_iter()
                    .rev()
                    .find(|mutation| !matches!(mutation, Mutation::Write(_)))
                    .filter(|mutation| matches!(mutation, Mutation::Remove))
                    .filter(|_| self.storage.contains_key(key))
            })
            .count();
        n + p - q
    }

    fn iter<'a>(&'a self) -> Box<dyn 'a + Iterator<Item = (E::Key, &E)>> {
        let keys: HashSet<_, FnvBuildHasher> = self.log.keys().collect();
        Box::new(
            self.storage
                .iter()
                .filter(move |(key, _)| !keys.contains(key))
                .chain(self.log.pairs().flat_map(move |(key, mut entry)| {
                    entry
                        .next_back()
                        .and_then(|mutation| mutation.as_entity().map(|entity| (*key, entity)))
                })),
        )
    }

    // This does not require logging, because only keys and user data are
    // exposed. Items yielded by this iterator are not recorded as writes.
    fn iter_mut<'a>(&'a mut self) -> Box<dyn 'a + Iterator<Item = (E::Key, &mut E::Data)>>
    where
        E: Payload,
    {
        let keys: HashSet<_, FnvBuildHasher> = self.log.keys().cloned().collect();
        Box::new(
            self.storage
                .iter_mut()
                .filter(move |(key, _)| !keys.contains(key))
                .chain(self.log.pairs_mut().flat_map(move |(key, mut entry)| {
                    entry.next_back().and_then(|mutation| {
                        mutation
                            .as_entity_mut()
                            .map(|entity| (*key, entity.get_mut()))
                    })
                })),
        )
    }
}

impl<T, E> From<T> for Journaled<T, E>
where
    T: Default + Dispatch<E> + JournalState + Storage<E> + Unjournaled,
    E: Entity<Storage = T>,
{
    fn from(storage: T) -> Self {
        Journaled::transact(storage)
    }
}

impl<T, E> Get<E> for Journaled<T, E>
where
    T: Default + Dispatch<E> + JournalState + Storage<E> + Unjournaled,
    E: Entity<Storage = T>,
{
    fn get(&self, key: &E::Key) -> Option<&E> {
        if let Some(mutation) = self.log.get_all(key).next_back() {
            mutation.as_entity()
        }
        else {
            self.storage.get(key)
        }
    }

    fn get_mut(&mut self, key: &E::Key) -> Option<&mut E> {
        let entity = self.get(key).cloned();
        if let Some(entity) = entity {
            self.log.append(*key, Mutation::Write(entity));
            if let Mutation::Write(ref mut entity) = self.log.back_mut().unwrap().1 {
                Some(entity)
            }
            else {
                unreachable!()
            }
        }
        else {
            None
        }
    }
}

impl<T, E> Insert<E> for Journaled<T, E>
where
    T: Default + Dispatch<E> + Insert<E> + JournalState + Storage<E> + Unjournaled,
    E: Entity<Storage = T>,
    E::Key: SyntheticKey<T::State>,
{
    fn insert(&mut self, entity: E) -> E::Key {
        let key = SyntheticKey::synthesize(&mut self.state);
        self.log.append(key, Mutation::Insert(entity));
        key
    }
}

impl<T, E> InsertWithKey<E> for Journaled<T, E>
where
    T: Default + Dispatch<E> + InsertWithKey<E> + JournalState + Storage<E> + Unjournaled,
    E: Entity<Storage = T>,
{
    fn insert_with_key(&mut self, key: &E::Key, entity: E) -> Option<E> {
        let occupant = self.get(key).cloned();
        self.log
            .append(*key, Mutation::Insert(entity))
            .and_then(|| occupant)
    }
}

impl<T, E> Remove<E> for Journaled<T, E>
where
    T: Default + Dispatch<E> + JournalState + Storage<E> + Unjournaled,
    E: Entity<Storage = T>,
{
    fn remove(&mut self, key: &E::Key) -> Option<E> {
        let occupant = self.get(key).cloned();
        if occupant.is_some() {
            self.log.append(*key, Mutation::Remove);
        }
        occupant
    }
}

#[cfg(test)]
mod tests {
    use crate::entity::storage::tests::Node;
    use crate::entity::storage::{Enumerate, Get, Insert, Journaled, Remove, SlotEntityMap};

    #[test]
    fn insert_abort() {
        let mut storage = SlotEntityMap::default();
        let k1 = Insert::insert(&mut storage, Node::default());

        let mut storage = Journaled::transact(storage);
        let k2 = Insert::insert(&mut storage, Node::default());
        assert!(Get::get(&storage, &k1).is_some());
        assert!(Get::get(&storage, &k2).is_some());

        let storage = storage.abort();
        assert!(Get::get(&storage, &k1).is_some());
        assert!(Get::get(&storage, &k2).is_none());
    }

    // TODO: Write a similar test for dependent storage (using
    //       `commit_with_rekeying`).
    #[test]
    fn independent_insert_commit() {
        let mut storage = SlotEntityMap::default();
        let k1 = Insert::insert(&mut storage, Node::default());

        let mut storage = Journaled::transact(storage);
        let k2 = Insert::insert(&mut storage, Node::default());
        assert!(Get::get(&storage, &k1).is_some());
        assert!(Get::get(&storage, &k2).is_some());

        let (storage, rekeying) = storage.commit_and_rekey();
        let k2 = rekeying.get(&k2).unwrap();
        assert!(Get::get(&storage, &k1).is_some());
        assert!(Get::get(&storage, &k2).is_some());
    }

    #[test]
    fn enumerate() {
        let mut storage = SlotEntityMap::default();
        let k1 = Insert::insert(&mut storage, Node::default());
        assert_eq!(1, Enumerate::len(&storage));

        let mut storage = Journaled::transact(storage);
        let k2 = Insert::insert(&mut storage, Node::default());
        let k3 = Insert::insert(&mut storage, Node::default());
        assert_eq!(3, Enumerate::len(&storage));

        // Remove an entity that is only present in the log.
        Get::get_mut(&mut storage, &k2).unwrap();
        Get::get_mut(&mut storage, &k3).unwrap();
        Remove::remove(&mut storage, &k2);
        assert_eq!(2, Enumerate::len(&storage));
        // Remove an entity that is only present in the underlying storage.
        Remove::remove(&mut storage, &k1);
        assert_eq!(1, Enumerate::len(&storage));

        let storage = storage.abort();
        assert_eq!(1, Enumerate::len(&storage));
    }
}
