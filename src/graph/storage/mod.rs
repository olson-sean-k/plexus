//! Storage for topological data in a mesh.

use fnv::FnvBuildHasher;
use std::collections::HashMap;
use std::hash::Hash;

use crate::graph::storage::convert::{AsStorage, AsStorageMut, FromInnerKey};
use crate::graph::topology::Topological;

pub mod convert;

pub trait KeySequence: Copy + Default + Sized {
    fn next_key(self) -> Self;
}

impl KeySequence for () {
    fn next_key(self) -> Self {}
}

impl KeySequence for u64 {
    fn next_key(self) -> Self {
        self + 1
    }
}

pub trait OpaqueKey: Copy + Eq + Hash + Sized {
    type Sequence: KeySequence;
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct VertexKey(u64);

impl FromInnerKey<u64> for VertexKey {
    fn from_inner_key(inner: u64) -> Self {
        VertexKey(inner)
    }
}

impl KeySequence for VertexKey {
    fn next_key(self) -> Self {
        let VertexKey(inner) = self;
        VertexKey(inner.next_key())
    }
}

impl OpaqueKey for VertexKey {
    type Sequence = Self;
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct EdgeKey(VertexKey, VertexKey);

impl EdgeKey {
    pub(in crate::graph) fn opposite(self) -> EdgeKey {
        let (a, b) = self.into();
        (b, a).into()
    }
}

impl From<(VertexKey, VertexKey)> for EdgeKey {
    fn from(key: (VertexKey, VertexKey)) -> Self {
        EdgeKey(key.0, key.1)
    }
}

impl Into<(VertexKey, VertexKey)> for EdgeKey {
    fn into(self) -> (VertexKey, VertexKey) {
        (self.0, self.1)
    }
}

impl OpaqueKey for EdgeKey {
    type Sequence = ();
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct FaceKey(u64);

impl FromInnerKey<u64> for FaceKey {
    fn from_inner_key(inner: u64) -> Self {
        FaceKey(inner)
    }
}

impl KeySequence for FaceKey {
    fn next_key(self) -> Self {
        let FaceKey(inner) = self;
        FaceKey(inner.next_key())
    }
}

impl OpaqueKey for FaceKey {
    type Sequence = Self;
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Key {
    Vertex(VertexKey),
    Edge(EdgeKey),
    Face(FaceKey),
}

impl From<VertexKey> for Key {
    fn from(key: VertexKey) -> Self {
        Key::Vertex(key)
    }
}

impl From<EdgeKey> for Key {
    fn from(key: EdgeKey) -> Self {
        Key::Edge(key)
    }
}

impl From<FaceKey> for Key {
    fn from(key: FaceKey) -> Self {
        Key::Face(key)
    }
}

#[derive(Clone, Default)]
pub struct Storage<T>
where
    T: Topological,
{
    sequence: <<T as Topological>::Key as OpaqueKey>::Sequence,
    hash: HashMap<T::Key, T, FnvBuildHasher>,
}

impl<T> Storage<T>
where
    T: Topological,
{
    pub fn new() -> Self {
        Storage {
            sequence: Default::default(),
            hash: HashMap::default(),
        }
    }

    // This function isn't strictly necessary, because `HashMap::new` begins
    // with a capacity of zero and does not allocate (and is used in
    // `Storage::new`). However, `Storage` abstracts its underlying data
    // structure, so the notion of an unallocated and empty container is
    // explicit.
    pub fn empty() -> Self {
        Self::new()
    }

    pub fn map_values_into<U, F>(self, mut f: F) -> Storage<U>
    where
        U: Topological<Key = T::Key>,
        F: FnMut(T) -> U,
    {
        let mut hash = HashMap::default();
        for (key, value) in self.hash {
            hash.insert(key, f(value));
        }
        Storage {
            sequence: self.sequence,
            hash,
        }
    }

    pub fn len(&self) -> usize {
        self.hash.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&T::Key, &T)> {
        self.hash.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&T::Key, &mut T)> {
        self.hash.iter_mut()
    }

    pub fn keys(&self) -> impl Iterator<Item = &T::Key> {
        self.hash.keys()
    }

    pub fn contains_key(&self, key: &T::Key) -> bool {
        self.hash.contains_key(key)
    }

    pub fn get(&self, key: &T::Key) -> Option<&T> {
        self.hash.get(key)
    }

    pub fn get_mut(&mut self, key: &T::Key) -> Option<&mut T> {
        self.hash.get_mut(key)
    }

    pub fn insert_with_key(&mut self, key: &T::Key, item: T) -> Option<T> {
        self.hash.insert(*key, item)
    }

    pub fn remove(&mut self, key: &T::Key) -> Option<T> {
        self.hash.remove(key)
    }
}

impl<T> Storage<T>
where
    T: Topological,
    T::Key: KeySequence + OpaqueKey<Sequence = T::Key>,
{
    pub fn insert(&mut self, item: T) -> T::Key {
        let key = self.sequence;
        self.hash.insert(key, item);
        self.sequence = self.sequence.next_key();
        key
    }
}

impl<T> AsStorage<T> for Storage<T>
where
    T: Topological,
{
    fn as_storage(&self) -> &Storage<T> {
        self
    }
}

impl<T> AsStorageMut<T> for Storage<T>
where
    T: Topological,
{
    fn as_storage_mut(&mut self) -> &mut Storage<T> {
        self
    }
}
