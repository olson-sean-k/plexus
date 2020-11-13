use std::collections::HashMap;
use std::convert::TryFrom;
use std::hash::{BuildHasher, Hash};

pub trait Rekeying {
    type Key: Copy + Eq;

    fn insert<T>(&mut self, from: T, to: T) -> Option<Self::Key>
    where
        Self::Key: From<T>;

    fn get_and_rekey<T>(&self, target: &mut T) -> bool
    where
        T: Copy + TryFrom<Self::Key>,
        Self::Key: From<T>;

    fn get_and_rekey_some<T>(&self, target: &mut Option<T>) -> bool
    where
        T: Copy + TryFrom<Self::Key>,
        Self::Key: From<T>,
    {
        if let Some(ref mut key) = target {
            self.get_and_rekey(key)
        }
        else {
            false
        }
    }
}

impl<K, H> Rekeying for HashMap<K, K, H>
where
    K: Copy + Eq + Hash,
    H: BuildHasher,
{
    type Key = K;

    fn insert<T>(&mut self, from: T, to: T) -> Option<Self::Key>
    where
        Self::Key: From<T>,
    {
        self.insert(from.into(), to.into())
    }

    fn get_and_rekey<T>(&self, target: &mut T) -> bool
    where
        T: Copy + TryFrom<Self::Key>,
        Self::Key: From<T>,
    {
        // TODO: This silently discards mismapped keys. Consider a map of
        //       `GraphKey`s where the key is a vertex key and the value is a
        //       face key.
        if let Some(key) = self
            .get(&target.clone().into())
            .and_then(|key| T::try_from(*key).ok())
        {
            *target = key;
            true
        }
        else {
            false
        }
    }
}

pub trait Rekey<K>
where
    K: Copy + Eq,
{
    fn rekey(&mut self, rekeying: &impl Rekeying<Key = K>) -> bool;
}
