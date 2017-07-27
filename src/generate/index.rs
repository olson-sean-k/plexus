use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

use generate::decompose::IntoVertices;
use generate::topology::Topological;

pub trait Indexer<T, K>
where
    T: IntoVertices + Topological,
{
    fn index<F>(&mut self, vertex: T::Vertex, f: F) -> (usize, Option<T::Vertex>)
    where
        F: Fn(&T::Vertex) -> &K;
}

pub struct HashIndexer<T, K>
where
    T: IntoVertices + Topological,
    K: Clone + Eq + Hash,
{
    hash: HashMap<K, usize>,
    n: usize,
    phantom: PhantomData<T>,
}

impl<T, K> HashIndexer<T, K>
where
    T: IntoVertices + Topological,
    K: Clone + Eq + Hash,
{
    fn new() -> Self {
        HashIndexer {
            hash: HashMap::new(),
            n: 0,
            phantom: PhantomData,
        }
    }
}

impl<T, K> Default for HashIndexer<T, K>
where
    T: IntoVertices + Topological,
    K: Clone + Eq + Hash,
{
    fn default() -> Self {
        HashIndexer::new()
    }
}

impl<T, K> Indexer<T, K> for HashIndexer<T, K>
where
    T: IntoVertices + Topological,
    K: Clone + Eq + Hash,
{
    fn index<F>(&mut self, input: T::Vertex, f: F) -> (usize, Option<T::Vertex>)
    where
        F: Fn(&T::Vertex) -> &K,
    {
        let mut vertex = None;
        let mut n = self.n;
        let index = self.hash.entry(f(&input).clone()).or_insert_with(|| {
            vertex = Some(input);
            let m = n;
            n += 1;
            m
        });
        self.n = n;
        (*index, vertex)
    }
}

pub trait IndexTopology<T>: Sized
where
    T: IntoVertices + Topological,
{
    fn index_topology_with_key<N, K, F>(self, indexer: N, f: F) -> (Vec<usize>, Vec<T::Vertex>)
    where
        N: Indexer<T, K>,
        F: Fn(&T::Vertex) -> &K;

    fn index_topology<N>(self, indexer: N) -> (Vec<usize>, Vec<T::Vertex>)
    where
        N: Indexer<T, T::Vertex>,
    {
        self.index_topology_with_key::<N, T::Vertex, _>(indexer, |vertex| vertex)
    }
}

impl<T, I> IndexTopology<T> for I
where
    I: Iterator<Item = T>,
    T: IntoVertices + Topological,
{
    fn index_topology_with_key<N, K, F>(self, mut indexer: N, f: F) -> (Vec<usize>, Vec<T::Vertex>)
    where
        N: Indexer<T, K>,
        F: Fn(&T::Vertex) -> &K,
    {
        let mut indeces = Vec::new();
        let mut vertices = Vec::new();
        for topology in self {
            for vertex in topology.into_vertices() {
                let (index, vertex) = indexer.index(vertex, &f);
                indeces.push(index);
                if let Some(vertex) = vertex {
                    vertices.push(vertex);
                }
            }
        }
        (indeces, vertices)
    }
}
