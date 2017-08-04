use num::{Integer, Unsigned};
use std::marker::PhantomData;

pub trait Vertex {
    type Position;
    type Color;
    type Normal;
    type Texture;
}

pub trait IntoVertex<V>
where
    V: Vertex,
{
    fn into_vertex(self) -> (V::Position, V::Color, V::Normal, V::Texture);
}

pub struct DisjointBuffer<N, V>
where
    N: Integer + Unsigned,
    V: Vertex,
{
    phantom: PhantomData<(N, V)>, // TODO: Remove.
}
