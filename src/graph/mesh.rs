use itertools::Itertools;
use std::hash::Hash;
use std::iter::FromIterator;

use generate::{HashIndexer, IndexVertices, IntoTriangles, IntoVertices, Topological, Triangulate};
use graph::geometry::{Attribute, Geometry};
use graph::storage::{EdgeKey, FaceKey, Key, OpaqueKey, Storage, VertexKey};

#[derive(Clone, Debug)]
pub struct Vertex<T, K>
where
    T: Attribute,
    K: Key,
{
    pub geometry: T,
    pub(super) edge: Option<EdgeKey<K>>,
}

impl<T, K> Vertex<T, K>
where
    T: Attribute,
    K: Key,
{
    fn new() -> Self {
        Vertex::with_geometry(T::default())
    }

    fn with_geometry(geometry: T) -> Self {
        Vertex {
            geometry: geometry,
            edge: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Edge<T, K>
where
    T: Attribute,
    K: Key,
{
    pub geometry: T,
    pub(super) opposite: Option<EdgeKey<K>>,
    pub(super) next: Option<EdgeKey<K>>,
    pub(super) vertex: VertexKey<K>,
}

impl<T, K> Edge<T, K>
where
    T: Attribute,
    K: Key,
{
    fn new(vertex: VertexKey<K>) -> Self {
        Edge::with_geometry(vertex, T::default())
    }

    fn with_geometry(vertex: VertexKey<K>, geometry: T) -> Self {
        Edge {
            geometry: geometry,
            opposite: None,
            next: None,
            vertex: vertex,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Face<T, K>
where
    T: Attribute,
    K: Key,
{
    pub geometry: T,
    pub(super) edge: EdgeKey<K>,
}

impl<T, K> Face<T, K>
where
    T: Attribute,
    K: Key,
{
    fn new(edge: EdgeKey<K>) -> Self {
        Face::with_geometry(edge, T::default())
    }

    fn with_geometry(edge: EdgeKey<K>, geometry: T) -> Self {
        Face {
            geometry: geometry,
            edge: edge,
        }
    }
}

pub struct Mesh<G, K = u64>
where
    G: Geometry,
    K: Key,
{
    pub(super) vertices: Storage<K, Vertex<G::Vertex, K>>,
    pub(super) edges: Storage<K, Edge<G::Edge, K>>,
    pub(super) faces: Storage<K, Face<G::Face, K>>,
}

impl<G, K> Mesh<G, K>
where
    G: Geometry,
    K: Key,
{
    pub fn new() -> Self {
        Mesh {
            vertices: Storage::new(),
            edges: Storage::new(),
            faces: Storage::new(),
        }
    }

    fn insert_vertex(&mut self, geometry: G::Vertex) -> VertexKey<K> {
        self.vertices.insert(Vertex::with_geometry(geometry)).into()
    }

    fn insert_edge(
        &mut self,
        vertices: (VertexKey<K>, VertexKey<K>),
        geometry: G::Edge,
    ) -> Result<EdgeKey<K>, ()> {
        let (i, j) = vertices;
        let ij = self.edges.insert(Edge::with_geometry(j, geometry));
        self.vertices.get_mut(&i.to_inner()).unwrap().edge = Some(ij.into());
        Ok(ij.into())
    }

    fn insert_triangle(
        &mut self,
        edges: (EdgeKey<K>, EdgeKey<K>, EdgeKey<K>),
        geometry: G::Face,
    ) -> Result<FaceKey<K>, ()> {
        let (ij, jk, ki) = edges;
        // Link each half-edge with its next half-edge.
        self.edges.get_mut(&ij.to_inner()).unwrap().next = Some(jk.into());
        self.edges.get_mut(&jk.to_inner()).unwrap().next = Some(ki.into());
        self.edges.get_mut(&ki.to_inner()).unwrap().next = Some(ij.into());
        Ok(
            self.faces
                .insert(Face::with_geometry(ij.into(), geometry))
                .into(),
        )
    }
}

impl<G, K> AsRef<Mesh<G, K>> for Mesh<G, K>
where
    G: Geometry,
    K: Key,
{
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<G, K> AsMut<Mesh<G, K>> for Mesh<G, K>
where
    G: Geometry,
    K: Key,
{
    fn as_mut(&mut self) -> &mut Self {
        self
    }
}

impl<G, K, T> FromIterator<T> for Mesh<G, K>
where
    G: Geometry,
    K: Key,
    T: IntoTriangles + IntoVertices + Topological,
    T::Vertex: Eq + Hash + Into<G::Vertex>,
{
    fn from_iter<I>(input: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut mesh = Mesh::new();
        let (indeces, vertices) = input
            .into_iter()
            .triangulate()
            .index_vertices(HashIndexer::default());
        let vertices = vertices
            .into_iter()
            .map(|vertex| mesh.insert_vertex(vertex.into()))
            .collect::<Vec<_>>();
        for mut triangle in &indeces.into_iter().chunks(3) {
            // Map from the indeces into the original buffers to the keys
            // referring to the vertices in the mesh.
            let (i, j, k) = (
                vertices[triangle.next().unwrap()],
                vertices[triangle.next().unwrap()],
                vertices[triangle.next().unwrap()],
            );
            let (ij, jk, ki) = (
                mesh.insert_edge((i, j), G::Edge::default()).unwrap(),
                mesh.insert_edge((j, k), G::Edge::default()).unwrap(),
                mesh.insert_edge((k, i), G::Edge::default()).unwrap(),
            );
            mesh.insert_triangle((ij, jk, ki), G::Face::default());
        }
        mesh
    }
}
