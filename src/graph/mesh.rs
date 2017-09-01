use itertools::Itertools;
use std::hash::Hash;
use std::iter::FromIterator;

use generate::{HashIndexer, IndexVertices, IntoTriangles, IntoVertices, Topological, Triangulate};
use graph::geometry::{Attribute, Geometry};
use graph::storage::{EdgeKey, FaceKey, Storage, VertexKey};
use graph::topology::{FaceMut, FaceRef};

#[derive(Clone, Debug)]
pub struct Vertex<T>
where
    T: Attribute,
{
    pub geometry: T,
    pub(super) edge: Option<EdgeKey>,
}

impl<T> Vertex<T>
where
    T: Attribute,
{
    fn new(geometry: T) -> Self {
        Vertex {
            geometry: geometry,
            edge: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Edge<T>
where
    T: Attribute,
{
    pub geometry: T,
    pub(super) vertex: VertexKey,
    pub(super) opposite: Option<EdgeKey>,
    pub(super) next: Option<EdgeKey>,
    pub(super) face: Option<FaceKey>,
}

impl<T> Edge<T>
where
    T: Attribute,
{
    fn new(vertex: VertexKey, geometry: T) -> Self {
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
pub struct Face<T>
where
    T: Attribute,
{
    pub geometry: T,
    pub(super) edge: EdgeKey,
}

impl<T> Face<T>
where
    T: Attribute,
{
    fn new(edge: EdgeKey, geometry: T) -> Self {
        Face {
            geometry: geometry,
            edge: edge,
        }
    }
}

pub struct Mesh<G = ()>
where
    G: Geometry,
{
    pub(super) vertices: Storage<VertexKey, Vertex<G::Vertex>>,
    pub(super) edges: Storage<EdgeKey, Edge<G::Edge>>,
    pub(super) faces: Storage<FaceKey, Face<G::Face>>,
}

impl<G> Mesh<G>
where
    G: Geometry,
{
    pub fn new() -> Self {
        Mesh {
            vertices: Storage::new(),
            edges: Storage::new(),
            faces: Storage::new(),
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    pub fn face(&self, face: FaceKey) -> Option<FaceRef<G>> {
        self.faces.get(&face).map(|_| FaceRef::new(self, face))
    }

    pub fn face_mut(&mut self, face: FaceKey) -> Option<FaceMut<G>> {
        match self.faces.contains_key(&face) {
            true => Some(FaceMut::new(self, face)),
            _ => None,
        }
    }

    fn insert_vertex(&mut self, geometry: G::Vertex) -> VertexKey {
        self.vertices.insert_with_generator(Vertex::new(geometry))
    }

    fn insert_edge(
        &mut self,
        vertices: (VertexKey, VertexKey),
        geometry: G::Edge,
    ) -> Result<EdgeKey, ()> {
        let (a, b) = vertices;
        let ab = (a, b).into();
        let ba = (b, a).into();
        let mut edge = Edge::new(b, geometry);
        if let Some(opposite) = self.edges.get_mut(&ba) {
            edge.opposite = Some(ba);
            opposite.opposite = Some(ab);
        }
        self.edges.insert_with_key(&ab, edge);
        self.vertices.get_mut(&a).unwrap().edge = Some(ab);
        Ok(ab)
    }

    fn insert_triangle(
        &mut self,
        edges: (EdgeKey, EdgeKey, EdgeKey),
        geometry: G::Face,
    ) -> Result<FaceKey, ()> {
        let (ab, bc, ca) = edges;
        let face = self.faces.insert_with_generator(Face::new(ab, geometry));
        self.connect_edges_in_face(face, (ab, bc));
        self.connect_edges_in_face(face, (bc, ca));
        // TODO: Connecting these edges creates a cycle. This means code must
        //       be able to detect these cycles. Is this okay?
        self.connect_edges_in_face(face, (ca, ab));
        Ok(face)
    }

    fn connect_edges_in_face(&mut self, face: FaceKey, edges: (EdgeKey, EdgeKey)) {
        let edge = self.edges.get_mut(&edges.0).unwrap();
        edge.next = Some(edges.1);
        edge.face = Some(face);
    }
}

impl<G> AsRef<Mesh<G>> for Mesh<G>
where
    G: Geometry,
{
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<G> AsMut<Mesh<G>> for Mesh<G>
where
    G: Geometry,
{
    fn as_mut(&mut self) -> &mut Self {
        self
    }
}

impl<G, T> FromIterator<T> for Mesh<G>
where
    G: Geometry,
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

#[cfg(test)]
mod tests {
    use r32;
    use generate::*;
    use graph::*;

    #[test]
    fn collect_topology_into_mesh() {
        let mesh = sphere::UVSphere::<f32>::with_unit_radius(3, 2)
            .spatial_polygons() // 6 triangles, 18 vertices.
            .map_vertices(|(x, y, z)| (r32::from(x), r32::from(y), r32::from(z)))
            .triangulate()
            .collect::<Mesh<(r32, r32, r32)>>();

        assert_eq!(5, mesh.vertex_count());
        assert_eq!(18, mesh.edge_count());
        assert_eq!(6, mesh.face_count());
    }
}
