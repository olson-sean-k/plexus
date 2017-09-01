use std::collections::HashMap;
use std::hash::Hash;
use std::ops::{Deref, DerefMut};

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Key(u64);

impl Deref for Key {
    type Target = u64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Key {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub trait Generator: Copy + Default {
    fn next(&self) -> Self;
}

impl Generator for () {
    fn next(&self) -> Self {}
}

impl Generator for Key {
    fn next(&self) -> Self {
        Key(**self + 1)
    }
}

pub trait OpaqueKey {
    type Key: Copy + Eq + Hash;
    type Generator: Generator;

    fn to_inner(&self) -> Self::Key;
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct VertexKey(Key);

impl From<Key> for VertexKey {
    fn from(key: Key) -> Self {
        VertexKey(key)
    }
}

impl OpaqueKey for VertexKey {
    type Key = Key;
    type Generator = Key;

    fn to_inner(&self) -> Self::Key {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct EdgeKey(Key, Key);

impl OpaqueKey for EdgeKey {
    type Key = (Key, Key);
    type Generator = ();

    fn to_inner(&self) -> Self::Key {
        (self.0, self.1)
    }
}

impl From<(Key, Key)> for EdgeKey {
    fn from(key: (Key, Key)) -> Self {
        EdgeKey(key.0, key.1)
    }
}

impl From<(VertexKey, VertexKey)> for EdgeKey {
    fn from(key: (VertexKey, VertexKey)) -> Self {
        EdgeKey(key.0.to_inner(), key.1.to_inner())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct FaceKey(Key);

impl From<Key> for FaceKey {
    fn from(key: Key) -> Self {
        FaceKey(key)
    }
}

impl OpaqueKey for FaceKey {
    type Key = Key;
    type Generator = Key;

    fn to_inner(&self) -> Self::Key {
        self.0
    }
}

pub struct Storage<K, T>(K::Generator, HashMap<K::Key, T>)
where
    K: OpaqueKey;

impl<K, T> Storage<K, T>
where
    K: OpaqueKey,
{
    pub fn new() -> Self {
        Storage(K::Generator::default(), HashMap::new())
    }

    pub fn map_values<U, F>(self, mut f: F) -> Storage<K, U>
    where
        F: FnMut(T) -> U,
    {
        let mut hash = HashMap::new();
        for (key, value) in self.1 {
            hash.insert(key, f(value));
        }
        Storage(self.0, hash)
    }

    #[inline(always)]
    pub fn insert_with_key(&mut self, key: &K, item: T) {
        self.1.insert(key.to_inner(), item);
    }

    #[inline(always)]
    pub fn contains_key(&self, key: &K) -> bool {
        self.1.contains_key(&key.to_inner())
    }

    #[inline(always)]
    pub fn get(&self, key: &K) -> Option<&T> {
        self.1.get(&key.to_inner())
    }

    #[inline(always)]
    pub fn get_mut(&mut self, key: &K) -> Option<&mut T> {
        self.1.get_mut(&key.to_inner())
    }
}

impl<K, T> Storage<K, T>
where
    K: From<Key> + OpaqueKey<Key = Key, Generator = Key>,
{
    pub fn insert_with_generator(&mut self, item: T) -> K {
        let key = self.0;
        self.1.insert(key, item);
        self.0 = self.0.next();
        key.into()
    }
}

impl<K, T> Deref for Storage<K, T>
where
    K: OpaqueKey,
{
    type Target = HashMap<K::Key, T>;

    fn deref(&self) -> &Self::Target {
        &self.1
    }
}

impl<K, T> DerefMut for Storage<K, T>
where
    K: OpaqueKey,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.1
    }
}
