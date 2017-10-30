/// Storage for topological data in a mesh.

use std::collections::HashMap;
use std::collections::hash_map;
use std::hash::Hash;
use std::ops::{Deref, DerefMut};

use graph::topology::Topological;

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

pub trait OpaqueKey: Sized {
    type RawKey: Copy + Eq + Hash + Into<Self>;
    type Generator: Generator;

    fn to_inner(&self) -> Self::RawKey;
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct VertexKey(Key);

impl From<Key> for VertexKey {
    fn from(key: Key) -> Self {
        VertexKey(key)
    }
}

impl OpaqueKey for VertexKey {
    type RawKey = Key;
    type Generator = Key;

    fn to_inner(&self) -> Self::RawKey {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct EdgeKey(Key, Key);

impl EdgeKey {
    pub(crate) fn to_vertex_keys(&self) -> (VertexKey, VertexKey) {
        (self.0.into(), self.1.into())
    }
}

impl OpaqueKey for EdgeKey {
    type RawKey = (Key, Key);
    type Generator = ();

    fn to_inner(&self) -> Self::RawKey {
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
    type RawKey = Key;
    type Generator = Key;

    fn to_inner(&self) -> Self::RawKey {
        self.0
    }
}

pub type StorageIter<'a, T> = hash_map::Iter<'a, <<T as Topological>::Key as OpaqueKey>::RawKey, T>;
pub type StorageIterMut<'a, T> = hash_map::IterMut<
    'a,
    <<T as Topological>::Key as OpaqueKey>::RawKey,
    T,
>;

pub struct Storage<T>(
    <<T as Topological>::Key as OpaqueKey>::Generator,
    HashMap<<<T as Topological>::Key as OpaqueKey>::RawKey, T>,
)
where
    T: Topological;

impl<T> Storage<T>
where
    T: Topological,
{
    pub fn new() -> Self {
        Storage(
            <<T as Topological>::Key as OpaqueKey>::Generator::default(),
            HashMap::new(),
        )
    }

    pub fn map_values_into<U, F>(self, mut f: F) -> Storage<U>
    where
        U: Topological<Key = T::Key>,
        F: FnMut(T) -> U,
    {
        let mut hash = HashMap::new();
        for (key, value) in self.1 {
            hash.insert(key, f(value));
        }
        Storage(self.0, hash)
    }

    #[inline(always)]
    pub fn insert_with_key(&mut self, key: &T::Key, item: T) {
        self.1.insert(key.to_inner(), item);
    }

    #[inline(always)]
    pub fn contains_key(&self, key: &T::Key) -> bool {
        self.1.contains_key(&key.to_inner())
    }

    #[inline(always)]
    pub fn get(&self, key: &T::Key) -> Option<&T> {
        self.1.get(&key.to_inner())
    }

    #[inline(always)]
    pub fn get_mut(&mut self, key: &T::Key) -> Option<&mut T> {
        self.1.get_mut(&key.to_inner())
    }

    #[inline(always)]
    pub fn remove(&mut self, key: &T::Key) -> Option<T> {
        self.1.remove(&key.to_inner())
    }
}

impl<T> Storage<T>
where
    T: Topological,
    T::Key: From<Key> + OpaqueKey<RawKey = Key, Generator = Key>,
{
    pub fn insert_with_generator(&mut self, item: T) -> T::Key {
        let key = self.0;
        self.1.insert(key, item);
        self.0 = self.0.next();
        key.into()
    }
}

impl<T> Deref for Storage<T>
where
    T: Topological,
{
    type Target = HashMap<<<T as Topological>::Key as OpaqueKey>::RawKey, T>;

    fn deref(&self) -> &Self::Target {
        &self.1
    }
}

impl<T> DerefMut for Storage<T>
where
    T: Topological,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.1
    }
}
