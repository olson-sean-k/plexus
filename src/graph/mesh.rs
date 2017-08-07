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
    pub(super) vertex: VertexKey<K>,
    pub(super) opposite: Option<EdgeKey<K>>,
    pub(super) next: Option<EdgeKey<K>>,
    pub(super) face: Option<FaceKey<K>>,
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
            vertex: vertex,
            opposite: None,
            next: None,
            face: None,
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
        let (a, b) = vertices;
        let ab = self.edges.insert(Edge::with_geometry(b, geometry));
        self.vertices.get_mut(&a.to_inner()).unwrap().edge = Some(ab.into());
        Ok(ab.into())
    }

    fn connect_edges_in_face(&mut self, face: FaceKey<K>, edges: (EdgeKey<K>, EdgeKey<K>)) {
        let edge = self.edges.get_mut(&edges.0.to_inner()).unwrap();
        edge.next = edges.1.into();
        edge.face = face.into();
    }

    fn insert_triangle(
        &mut self,
        edges: (EdgeKey<K>, EdgeKey<K>, EdgeKey<K>),
        geometry: G::Face,
    ) -> Result<FaceKey<K>, ()> {
        let (ab, bc, ca) = edges;
        let face = self.faces
            .insert(Face::with_geometry(ab.into(), geometry))
            .into();
        self.connect_edges_in_face(face, (ab, bc));
        self.connect_edges_in_face(face, (bc, ca));
        self.connect_edges_in_face(face, (ca, ab));
        Ok(face)
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
            let (a, b, c) = (
                vertices[triangle.next().unwrap()],
                vertices[triangle.next().unwrap()],
                vertices[triangle.next().unwrap()],
            );
            let (ab, bc, ca) = (
                mesh.insert_edge((a, b), G::Edge::default()).unwrap(),
                mesh.insert_edge((b, c), G::Edge::default()).unwrap(),
                mesh.insert_edge((c, a), G::Edge::default()).unwrap(),
            );
            mesh.insert_triangle((ab, bc, ca), G::Face::default())
                .unwrap();
        }
        mesh
    }
}
