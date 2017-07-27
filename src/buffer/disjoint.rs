use num::{Integer, Unsigned};
use std::marker::PhantomData;

pub trait Vertex {
    type PositionData;
    type ColorData;
    type NormalData;
    type TextureData;
}

pub trait IntoVertex<V>
where
    V: Vertex,
{
    fn into_vertex(self) -> (V::PositionData, V::ColorData, V::NormalData, V::TextureData);
}

pub struct DisjointBuffer<N, V>
where
    N: Integer + Unsigned,
    V: Vertex,
{
    phantom: PhantomData<(N, V)>, // TODO: Remove.
}
