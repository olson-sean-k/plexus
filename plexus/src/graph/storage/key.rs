//! Keys for graph storage.
//!
//! This module defines opaque keys for looking up graph data in storage. Keys
//! do not expose their underling data and enforce that lookups for a
//! particular topology use an appropriate key type.
//!
//! Unlike other topologies, arc keys expose some behaviors that reflect their
//! semantics. In particular, it is possible to convert an arc key into an
//! ordered pair of vertex keys representing the arc's _span_ and vice versa.
//! This allows for trivial and arbitrary queries of a graph's arcs, which
//! represent the fundemental topology of a graph.

use slotmap::DefaultKey;
use std::hash::Hash;

pub type InnerKey<K> = <K as OpaqueKey>::Inner;

pub trait FromInnerKey<K> {
    fn from_inner_key(key: K) -> Self;
}

pub trait IntoOpaqueKey<K> {
    fn into_opaque_key(self) -> K;
}

impl<K, I> IntoOpaqueKey<I> for K
where
    I: FromInnerKey<K>,
{
    fn into_opaque_key(self) -> I {
        I::from_inner_key(self)
    }
}

pub trait OpaqueKey: Copy + Eq + Hash + Sized {
    type Inner: Copy + Sized;

    fn from_inner(key: Self::Inner) -> Self;

    fn into_inner(self) -> Self::Inner;
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct VertexKey(DefaultKey);

impl OpaqueKey for VertexKey {
    type Inner = DefaultKey;

    fn from_inner(key: Self::Inner) -> Self {
        VertexKey(key)
    }

    fn into_inner(self) -> Self::Inner {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ArcKey(VertexKey, VertexKey);

impl ArcKey {
    pub(in crate::graph) fn into_opposite(self) -> ArcKey {
        let (a, b) = self.into();
        (b, a).into()
    }
}

impl From<(VertexKey, VertexKey)> for ArcKey {
    fn from(key: (VertexKey, VertexKey)) -> Self {
        ArcKey(key.0, key.1)
    }
}

impl Into<(VertexKey, VertexKey)> for ArcKey {
    fn into(self) -> (VertexKey, VertexKey) {
        (self.0, self.1)
    }
}

impl OpaqueKey for ArcKey {
    type Inner = (VertexKey, VertexKey);

    fn from_inner(key: Self::Inner) -> Self {
        ArcKey(key.0, key.1)
    }

    fn into_inner(self) -> Self::Inner {
        (self.0, self.1)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct EdgeKey(DefaultKey);

impl OpaqueKey for EdgeKey {
    type Inner = DefaultKey;

    fn from_inner(key: Self::Inner) -> Self {
        EdgeKey(key)
    }

    fn into_inner(self) -> Self::Inner {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct FaceKey(DefaultKey);

impl OpaqueKey for FaceKey {
    type Inner = DefaultKey;

    fn from_inner(key: Self::Inner) -> Self {
        FaceKey(key)
    }

    fn into_inner(self) -> Self::Inner {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Key {
    Vertex(VertexKey),
    Arc(ArcKey),
    Edge(EdgeKey),
    Face(FaceKey),
}

impl From<VertexKey> for Key {
    fn from(key: VertexKey) -> Self {
        Key::Vertex(key)
    }
}

impl From<ArcKey> for Key {
    fn from(key: ArcKey) -> Self {
        Key::Arc(key)
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
