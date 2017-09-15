use itertools::Itertools;
use std::hash::Hash;
use std::iter::FromIterator;

use generate::{self, FromIndexer, HashIndexer, IndexVertices, Indexer, IntoTriangles,
               IntoVertices, Triangle, Triangulate};
use graph::geometry::{FromGeometry, FromInteriorGeometry, Geometry, IntoGeometry,
                      IntoInteriorGeometry};
use graph::storage::{EdgeKey, FaceKey, Storage, VertexKey};
use graph::topology::{EdgeMut, EdgeRef, FaceMut, FaceRef, Topological, VertexMut, VertexRef};

#[derive(Clone, Debug)]
pub struct Vertex<G>
where
    G: Geometry,
{
    pub geometry: G::Vertex,
    pub(super) edge: Option<EdgeKey>,
}

impl<G> Vertex<G>
where
    G: Geometry,
{
    fn new(geometry: G::Vertex) -> Self {
        Vertex {
            geometry: geometry,
            edge: None,
        }
    }
}

impl<G, H> FromInteriorGeometry<Vertex<H>> for Vertex<G>
where
    G: Geometry,
    G::Vertex: FromGeometry<H::Vertex>,
    H: Geometry,
{
    fn from_interior_geometry(vertex: Vertex<H>) -> Self {
        Vertex {
            geometry: vertex.geometry.into_geometry(),
            edge: vertex.edge,
        }
    }
}

impl<G> Topological for Vertex<G>
where
    G: Geometry,
{
    type Key = VertexKey;
    type Attribute = G::Vertex;
}

#[derive(Clone, Debug)]
pub struct Edge<G>
where
    G: Geometry,
{
    pub geometry: G::Edge,
    pub(super) vertex: VertexKey,
    pub(super) opposite: Option<EdgeKey>,
    pub(super) next: Option<EdgeKey>,
    pub(super) face: Option<FaceKey>,
}

impl<G> Edge<G>
where
    G: Geometry,
{
    fn new(vertex: VertexKey, geometry: G::Edge) -> Self {
        Edge {
            geometry: geometry,
            vertex: vertex,
            opposite: None,
            next: None,
            face: None,
        }
    }
}

impl<G, H> FromInteriorGeometry<Edge<H>> for Edge<G>
where
    G: Geometry,
    G::Edge: FromGeometry<H::Edge>,
    H: Geometry,
{
    fn from_interior_geometry(edge: Edge<H>) -> Self {
        Edge {
            geometry: edge.geometry.into_geometry(),
            vertex: edge.vertex,
            opposite: edge.opposite,
            next: edge.next,
            face: edge.face,
        }
    }
}

impl<G> Topological for Edge<G>
where
    G: Geometry,
{
    type Key = EdgeKey;
    type Attribute = G::Edge;
}

#[derive(Clone, Debug)]
pub struct Face<G>
where
    G: Geometry,
{
    pub geometry: G::Face,
    pub(super) edge: EdgeKey,
}

impl<G> Face<G>
where
    G: Geometry,
{
    fn new(edge: EdgeKey, geometry: G::Face) -> Self {
        Face {
            geometry: geometry,
            edge: edge,
        }
    }
}

impl<G, H> FromInteriorGeometry<Face<H>> for Face<G>
where
    G: Geometry,
    G::Face: FromGeometry<H::Face>,
    H: Geometry,
{
    fn from_interior_geometry(face: Face<H>) -> Self {
        Face {
            geometry: face.geometry.into_geometry(),
            edge: face.edge,
        }
    }
}

impl<G> Topological for Face<G>
where
    G: Geometry,
{
    type Key = FaceKey;
    type Attribute = G::Face;
}

pub struct Mesh<G = ()>
where
    G: Geometry,
{
    pub(super) vertices: Storage<Vertex<G>>,
    pub(super) edges: Storage<Edge<G>>,
    pub(super) faces: Storage<Face<G>>,
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

    pub fn vertex(&self, vertex: VertexKey) -> Option<VertexRef<G>> {
        self.vertices
            .get(&vertex)
            .map(|_| VertexRef::new(self, vertex))
    }

    pub fn vertex_mut(&mut self, vertex: VertexKey) -> Option<VertexMut<G>> {
        match self.vertices.contains_key(&vertex) {
            true => Some(VertexMut::new(self, vertex)),
            _ => None,
        }
    }

    pub fn edge(&self, edge: EdgeKey) -> Option<EdgeRef<G>> {
        self.edges.get(&edge).map(|_| EdgeRef::new(self, edge))
    }

    pub fn edge_mut(&mut self, edge: EdgeKey) -> Option<EdgeMut<G>> {
        match self.edges.contains_key(&edge) {
            true => Some(EdgeMut::new(self, edge)),
            _ => None,
        }
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

    pub(crate) fn insert_vertex(&mut self, geometry: G::Vertex) -> VertexKey {
        self.vertices.insert_with_generator(Vertex::new(geometry))
    }

    pub(crate) fn insert_edge(
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

    pub(crate) fn insert_face(
        &mut self,
        edges: &[EdgeKey],
        geometry: G::Face,
    ) -> Result<FaceKey, ()> {
        if edges.len() < 3 {
            return Err(());
        }
        let face = self.faces
            .insert_with_generator(Face::new(edges[0], geometry));
        for index in 0..edges.len() {
            // TODO: Connecting these edges creates a cycle. This means code
            //       must be able to detect these cycles. Is this okay?
            self.connect_edges_in_face(face, (edges[index], edges[(index + 1) % edges.len()]));
        }
        Ok(face)
    }

    // TODO: This code orphans vertices; it does not remove vertices with no
    //       remaining associated edges. `FaceView::extrude` relies on this
    //       behavior.  Is this okay?
    pub(crate) fn remove_face(&mut self, face: FaceKey) -> Result<(), ()> {
        // Get all of the edges forming the face.
        let edges = {
            self.face(face)
                .unwrap()
                .edges()
                .map(|edge| edge.key)
                .collect::<Vec<_>>()
        };
        // For each edge, disconnect its opposite edge and originating vertex,
        // then remove it from the graph.
        for edge in edges {
            let (a, opposite) = {
                let edge = self.edges.get(&edge).unwrap();
                (edge.vertex, edge.opposite)
            };
            if let Some(edge) = opposite.map(|opposite| self.edges.get_mut(&opposite).unwrap()) {
                edge.opposite = None;
            }
            // Disconnect the originating vertex, if any.
            let vertex = self.vertices.get_mut(&a).unwrap();
            if vertex
                .edge
                .map(|outgoing| outgoing == edge)
                .unwrap_or(false)
            {
                vertex.edge = None;
            }
            self.edges.remove(&edge);
        }
        self.faces.remove(&face);
        Ok(())
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

impl<G, H> FromInteriorGeometry<Mesh<H>> for Mesh<G>
where
    G: Geometry,
    G::Vertex: FromGeometry<H::Vertex>,
    G::Edge: FromGeometry<H::Edge>,
    G::Face: FromGeometry<H::Face>,
    H: Geometry,
{
    fn from_interior_geometry(mesh: Mesh<H>) -> Self {
        let Mesh {
            vertices,
            edges,
            faces,
        } = mesh;
        // TODO: The new geometry should be recomputed or finalized here.
        Mesh {
            vertices: vertices.map_values_into(|vertex| vertex.into_interior_geometry()),
            edges: edges.map_values_into(|edge| edge.into_interior_geometry()),
            faces: faces.map_values_into(|face| face.into_interior_geometry()),
        }
    }
}

impl<G, P> FromIndexer<P, Triangle<P::Vertex>> for Mesh<G>
where
    G: Geometry,
    P: IntoTriangles + IntoVertices + generate::Topological,
    P::Vertex: IntoGeometry<G::Vertex>,
{
    fn from_indexer<I, N>(input: I, indexer: N) -> Self
    where
        I: IntoIterator<Item = P>,
        N: Indexer<Triangle<P::Vertex>, P::Vertex>,
    {
        let mut mesh = Mesh::new();
        let (indeces, vertices) = input.into_iter().triangulate().index_vertices(indexer);
        let vertices = vertices
            .into_iter()
            .map(|vertex| mesh.insert_vertex(vertex.into_geometry()))
            .collect::<Vec<_>>();
        for mut triangle in &indeces.into_iter().chunks(3) {
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
            mesh.insert_face(&[ab, bc, ca], G::Face::default()).unwrap();
        }
        mesh
    }
}

impl<G, P> FromIterator<P> for Mesh<G>
where
    G: Geometry,
    P: IntoTriangles + IntoVertices + generate::Topological,
    P::Vertex: Eq + Hash + IntoGeometry<G::Vertex>,
{
    fn from_iter<I>(input: I) -> Self
    where
        I: IntoIterator<Item = P>,
    {
        // TODO: This is fast and reliable, but the requirements on `P::Vertex`
        //       are difficult to achieve. Would `LruIndexer` be a better
        //       choice?
        Self::from_indexer(input, HashIndexer::default())
    }
}

#[cfg(test)]
mod tests {
    use generate::*;
    use graph::*;

    #[test]
    fn collect_topology_into_mesh() {
        let mesh = sphere::UVSphere::<f32>::with_unit_radius(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .triangulate()
            .collect_with_indexer::<Mesh<Triplet<f32>>, _>(LruIndexer::default());

        assert_eq!(5, mesh.vertex_count());
        assert_eq!(18, mesh.edge_count());
        assert_eq!(6, mesh.face_count());
    }
}
