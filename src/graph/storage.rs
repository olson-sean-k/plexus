use num::{Bounded, Integer};
use std::collections::HashMap;
use std::hash::Hash;
use std::ops::{Deref, DerefMut};

pub trait Key: Copy + Default + Eq + Hash + PartialEq {
    fn none() -> Self;
    fn next(&self) -> Self;

    #[inline(always)]
    fn is_none(&self) -> bool {
        *self == Self::none()
    }
}

impl<K> Key for K
where
    K: Bounded + Copy + Default + Eq + Hash + Integer + PartialEq,
{
    #[inline(always)]
    fn none() -> Self {
        K::max_value()
    }

    fn next(&self) -> Self {
        *self + Self::one()
    }
}

pub trait OpaqueKey<K>: From<K>
where
    K: Key,
{
    fn to_inner(&self) -> K;
}

macro_rules! opaque_key {
    ($($t:ident),*) => {$(
        #[derive(Copy, Clone, Debug)]
        pub struct $t<K>(K)
        where
            K: Key;

        impl<K> OpaqueKey<K> for $t<K>
        where
            K: Key,
        {
            fn to_inner(&self) -> K {
                self.0
            }
        }

        impl<K> From<K> for $t<K>
        where
            K: Key,
        {
            fn from(key: K) -> Self {
                $t(key)
            }
        }
    )*};
}
opaque_key!(VertexKey, EdgeKey, FaceKey);

pub struct Storage<K, T>(K, HashMap<K, T>)
where
    K: Key;

impl<K, T> Storage<K, T>
where
    K: Key,
{
    pub fn new() -> Self {
        Storage(K::default(), HashMap::new())
    }
}

impl<K, T> Storage<K, T>
where
    K: Key,
{
    pub fn insert(&mut self, item: T) -> K {
        let key = self.0;
        self.1.insert(key, item);
        self.0 = self.0.next();
        key
    }
}

impl<K, T> Deref for Storage<K, T>
where
    K: Key,
{
    type Target = HashMap<K, T>;

    fn deref(&self) -> &Self::Target {
        &self.1
    }
}

impl<K, T> DerefMut for Storage<K, T>
where
    K: Key,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.1
    }
}
