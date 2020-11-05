use slotmap::hop::HopSlotMap;
use slotmap::KeyData;
use std::convert::TryFrom;

use crate::entity::storage::journal::{JournalState, SyntheticKey, Unjournaled};
use crate::entity::storage::{
    AsStorage, AsStorageMut, Dispatch, Enumerate, Get, IndependentStorage, InnerKey, Insert, Key,
    Remove, StorageTarget,
};
use crate::entity::{Entity, Payload};

pub use slotmap::Key as SlotKey;

pub type SlotEntityMap<E> = HopSlotMap<InnerKey<<E as Entity>::Key>, E>;

pub struct State {
    floor: u32,
    index: u32,
    version: u32,
}

// See also the implementation of `JournalState` for `HopSlotMap`.
impl<K> SyntheticKey<State> for K
where
    K: Key,
    InnerKey<K>: SlotKey,
{
    // Key synthesis wraps the index within the interval [`floor`, `u32::MAX`].
    // Each overflow is carried into the version twice (the version is
    // incremented by two).
    fn synthesize(state: &mut State) -> Self {
        let State {
            ref floor,
            index,
            version,
        } = state;
        // This encoding depends on the implementation details of
        // `KeyData::from_ffi`. The dependency on `slotmap` is pinned to a
        // specific version; upgrading `slotmap` may require changing this code.
        // Note that the index is meaningful and is constructed in such a way
        // that synthetic keys never conflict with existing keys in the map.
        let synthetic = (u64::from(*version) << 32) | u64::from(*index);
        // An index of `u32::MAX` is considered a null key, but this condition
        // is not checked in `slotmap`.
        *index = match index.overflowing_add(1) {
            (index, false) => index,
            (_, true) => {
                *version = version.checked_add(2).expect("exhausted synthesized keys");
                *floor
            }
        };
        // `KeyData::from_ffi` is defensive and will modify version data such
        // that it is odd (representing an occupied slot).
        Key::from_inner(KeyData::from_ffi(synthetic).into())
    }
}

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

// See also the implementation of `SyntheticKey` for wrapped `SlotKey`s.
impl<E> JournalState for HopSlotMap<InnerKey<E::Key>, E>
where
    E: Entity,
    InnerKey<E::Key>: SlotKey,
{
    type State = State;

    // This state is used solely for key synthesis. Each key contains index and
    // version data. The state is based on the state of a slot map (namely its
    // capacity), which is used to synthesize keys that do not conflict with
    // existing keys in the map.
    fn state(&self) -> Self::State {
        // TODO: Is this recoverable? Is it useful to propagate such an error?
        let floor = u32::try_from(self.capacity())
            .unwrap()
            .checked_add(1)
            .expect("insufficient capacity for journaling");
        let index = floor;
        let version = 1;
        State {
            floor,
            index,
            version,
        }
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

impl<K, T> Unjournaled for HopSlotMap<K, T>
where
    K: SlotKey,
    T: Copy,
{
}

#[cfg(test)]
mod tests {
    use crate::entity::storage::journal::SyntheticKey;
    use crate::entity::storage::slot::State;
    use crate::entity::storage::tests::NodeKey;

    // TODO: This test exercises implementation details. Is there a better way
    //       to test key synthesis in general?
    #[test]
    fn synthetic_key_index_overflow() {
        let mut state = State {
            floor: u32::MAX - 1,
            index: u32::MAX - 1,
            version: 1,
        };

        let _ = NodeKey::synthesize(&mut state);
        assert_eq!(u32::MAX, state.index);
        assert_eq!(1, state.version);

        let _ = NodeKey::synthesize(&mut state);
        assert_eq!(u32::MAX - 1, state.index);
        assert_ne!(1, state.version);
    }
}
