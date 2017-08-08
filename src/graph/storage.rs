use num::Integer;
use std::collections::HashMap;
use std::hash::Hash;

pub trait Key: Copy + Default + Eq + Hash + PartialEq {}

impl<K> Key for K
where
    K: Copy + Default + Eq + Hash + PartialEq,
{
}

pub trait AtomicKey: Key {
    fn next(&self) -> Self;
}

impl<K> AtomicKey for K
where
    K: Integer + Key,
{
    fn next(&self) -> Self {
        *self + Self::one()
    }
}

pub trait OpaqueKey {
    type Key: Key;

    fn to_inner(&self) -> Self::Key;
}

macro_rules! opaque_key {
    ($($t:ident => $k:ident for $i:ty),*) => {$(
        #[derive(Copy, Clone, Debug)]
        pub struct $t<$k>($i)
        where
            $k: Key;

        impl<$k> OpaqueKey for $t<$k>
        where
            $k: Key,
        {
            type Key = $i;

            fn to_inner(&self) -> Self::Key {
                self.0
            }
        }

        impl<$k> From<$i> for $t<$k>
        where
            $k: Key,
        {
            fn from(key: $i) -> Self {
                $t(key)
            }
        }
    )*};
}
opaque_key!(VertexKey => K for K, EdgeKey => K for (K, K), FaceKey => K for K);

impl<K> From<(VertexKey<K>, VertexKey<K>)> for EdgeKey<K>
where
    K: Key,
{
    fn from(key: (VertexKey<K>, VertexKey<K>)) -> Self {
        EdgeKey((key.0.to_inner(), key.1.to_inner()))
    }
}

pub struct Storage<K, T>(K::Key, HashMap<K::Key, T>)
where
    K: OpaqueKey;

impl<K, T> Storage<K, T>
where
    K: OpaqueKey,
{
    pub fn new() -> Self {
        Storage(K::Key::default(), HashMap::new())
    }

    pub fn insert_with_key(&mut self, key: K, item: T) {
        self.1.insert(key.to_inner(), item);
    }

    pub fn get(&self, key: K) -> Option<&T> {
        self.1.get(&key.to_inner())
    }

    pub fn get_mut(&mut self, key: K) -> Option<&mut T> {
        self.1.get_mut(&key.to_inner())
    }
}

impl<K, T> Storage<K, T>
where
    K: OpaqueKey,
    K::Key: AtomicKey + Into<K>,
{
    pub fn insert(&mut self, item: T) -> K {
        let key = self.0;
        self.1.insert(key, item);
        self.0 = self.0.next();
        key.into()
    }
}
