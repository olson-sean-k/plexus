use num::{Integer, NumCast, Unsigned};

pub struct ConjointBuffer<N, V>
where
    N: Integer + Unsigned,
{
    indeces: Vec<N>,
    vertices: Vec<V>,
}

impl<N, V> ConjointBuffer<N, V>
where
    N: Integer + Unsigned,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn insert_index<T>(&mut self, index: T)
    where
        T: Into<N>,
    {
        self.indeces.push(index.into());
    }

    pub(crate) fn insert_vertex<T>(&mut self, vertex: T)
    where
        T: Into<V>,
    {
        self.vertices.push(vertex.into());
    }
}

impl<N, V> ConjointBuffer<N, V>
where
    N: Copy + Integer + NumCast + Unsigned,
{
    pub fn append(&mut self, other: &mut Self) {
        let offset = N::from(self.vertices.len()).unwrap();
        self.vertices.append(&mut other.vertices);
        self.indeces
            .extend(other.indeces.drain(..).map(|index| index + offset))
    }
}

impl<N, V> Default for ConjointBuffer<N, V>
where
    N: Integer + Unsigned,
{
    fn default() -> Self {
        ConjointBuffer {
            indeces: vec![],
            vertices: vec![],
        }
    }
}
