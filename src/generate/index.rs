use std::cmp;
use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

use generate::decompose::IntoVertices;
use generate::topology::{Arity, Topological};

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

pub struct LruIndexer<T, K>
where
    T: IntoVertices + Topological,
    K: Clone + PartialEq,
{
    lru: Vec<(K, usize)>,
    capacity: usize,
    n: usize,
    phantom: PhantomData<T>,
}

impl<T, K> LruIndexer<T, K>
where
    T: IntoVertices + Topological,
    K: Clone + PartialEq,
{
    pub fn new() -> Self {
        LruIndexer::with_capacity(16)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = cmp::max(1, capacity);
        LruIndexer {
            lru: Vec::with_capacity(capacity),
            capacity: capacity,
            n: 0,
            phantom: PhantomData,
        }
    }

    fn find(&self, key: &K) -> Option<(usize, usize)> {
        self.lru
            .iter()
            .enumerate()
            .find(|&(_, ref entry)| entry.0 == *key)
            .map(|(index, ref entry)| (index, entry.1))
    }
}

impl<T, K> Default for LruIndexer<T, K>
where
    T: IntoVertices + Topological,
    K: Clone + PartialEq,
{
    fn default() -> Self {
        LruIndexer::new()
    }
}

impl<T, K> Indexer<T, K> for LruIndexer<T, K>
where
    T: IntoVertices + Topological,
    K: Clone + PartialEq,
{
    fn index<F>(&mut self, input: T::Vertex, f: F) -> (usize, Option<T::Vertex>)
    where
        F: Fn(&T::Vertex) -> &K,
    {
        let mut vertex = None;
        let key = f(&input).clone();
        let index = if let Some(entry) = self.find(&key) {
            let vertex = self.lru.remove(entry.0);
            self.lru.push(vertex);
            entry.1
        }
        else {
            vertex = Some(input);
            let m = self.n;
            self.n += 1;
            if self.lru.len() >= self.capacity {
                self.lru.remove(0);
            }
            self.lru.push((key, m));
            m
        };
        (index, vertex)
    }
}

pub trait IndexVertices<T>: Sized
where
    T: Arity + IntoVertices + Topological,
{
    fn index_vertices_with_key<N, K, F>(self, indexer: N, f: F) -> (Vec<usize>, Vec<T::Vertex>)
    where
        N: Indexer<T, K>,
        F: Fn(&T::Vertex) -> &K;

    fn index_vertices<N>(self, indexer: N) -> (Vec<usize>, Vec<T::Vertex>)
    where
        N: Indexer<T, T::Vertex>,
    {
        self.index_vertices_with_key::<N, T::Vertex, _>(indexer, |vertex| vertex)
    }
}

impl<T, I> IndexVertices<T> for I
where
    I: Iterator<Item = T>,
    T: Arity + IntoVertices + Topological,
{
    fn index_vertices_with_key<N, K, F>(self, mut indexer: N, f: F) -> (Vec<usize>, Vec<T::Vertex>)
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

pub trait FromIndexer<P, Q>
where
    P: IntoVertices + Topological,
    Q: IntoVertices + Topological<Vertex = P::Vertex>,
{
    fn from_indexer<I, N>(input: I, indexer: N) -> Self
    where
        I: IntoIterator<Item = P>,
        N: Indexer<Q, P::Vertex>;
}

pub trait CollectWithIndexer<P, Q>
where
    P: IntoVertices + Topological,
    Q: IntoVertices + Topological<Vertex = P::Vertex>,
{
    fn collect_with_indexer<T, N>(self, indexer: N) -> T
    where
        T: FromIndexer<P, Q>,
        N: Indexer<Q, P::Vertex>;
}

impl<P, Q, I> CollectWithIndexer<P, Q> for I
where
    I: Iterator<Item = P>,
    P: IntoVertices + Topological,
    Q: IntoVertices + Topological<Vertex = P::Vertex>,
{
    fn collect_with_indexer<T, N>(self, indexer: N) -> T
    where
        T: FromIndexer<P, Q>,
        N: Indexer<Q, P::Vertex>,
    {
        T::from_indexer(self, indexer)
    }
}
