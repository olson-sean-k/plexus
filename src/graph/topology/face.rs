use std::marker::PhantomData;
use std::ops::{Add, Deref, DerefMut, Mul};

use geometry::Geometry;
use geometry::convert::AsPosition;
use graph::VecExt;
use graph::geometry::{FaceCentroid, FaceNormal};
use graph::geometry::alias::{ScaledFaceNormal, VertexPosition};
use graph::mesh::{Edge, Face, Mesh, Vertex};
use graph::storage::{EdgeKey, FaceKey, VertexKey};
use graph::topology::{EdgeKeyTopology, EdgeView, OrphanEdgeView, OrphanVertexView, OrphanView,
                      Topological, VertexView, View};

pub struct FaceView<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    mesh: M,
    key: FaceKey,
    phantom: PhantomData<G>,
}

impl<M, G> FaceView<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    pub(crate) fn new(mesh: M, face: FaceKey) -> Self {
        FaceView {
            mesh: mesh,
            key: face,
            phantom: PhantomData,
        }
    }

    pub fn key(&self) -> FaceKey {
        self.key
    }

    pub fn to_key_topology(&self) -> FaceKeyTopology {
        FaceKeyTopology::new(self.key, self.edges().map(|edge| edge.to_key_topology()))
    }

    pub fn arity(&self) -> usize {
        self.edges().count()
    }

    pub fn vertices(&self) -> VertexCirculator<&Mesh<G>, G> {
        VertexCirculator::from_edge_circulator(self.edges())
    }

    pub fn edges(&self) -> EdgeCirculator<&Mesh<G>, G> {
        EdgeCirculator::new(self.with_mesh_ref())
    }

    pub fn faces(&self) -> FaceCirculator<&Mesh<G>, G> {
        FaceCirculator::from_edge_circulator(self.edges())
    }

    // Resolve the `M` parameter to a concrete reference.
    fn with_mesh_ref(&self) -> FaceView<&Mesh<G>, G> {
        FaceView::new(self.mesh.as_ref(), self.key)
    }
}

impl<M, G> FaceView<M, G>
where
    M: AsRef<Mesh<G>> + AsMut<Mesh<G>>,
    G: Geometry,
{
    pub fn vertices_mut(&mut self) -> VertexCirculator<&mut Mesh<G>, G> {
        VertexCirculator::from_edge_circulator(self.edges_mut())
    }

    pub fn edges_mut(&mut self) -> EdgeCirculator<&mut Mesh<G>, G> {
        EdgeCirculator::new(self.with_mesh_mut())
    }

    pub fn faces_mut(&mut self) -> FaceCirculator<&mut Mesh<G>, G> {
        FaceCirculator::from_edge_circulator(self.edges_mut())
    }

    fn remove(self) -> Result<M, ()> {
        let FaceView { mut mesh, key, .. } = self;
        mesh.as_mut().remove_face(key)?;
        Ok(mesh)
    }

    // Resolve the `M` parameter to a concrete reference.
    fn with_mesh_mut(&mut self) -> FaceView<&mut Mesh<G>, G> {
        FaceView::new(self.mesh.as_mut(), self.key)
    }
}

impl<M, G> FaceView<M, G>
where
    M: AsRef<Mesh<G>> + AsMut<Mesh<G>>,
    G: FaceCentroid<Centroid = <G as Geometry>::Vertex> + Geometry,
{
    pub fn triangulate(self) -> Result<Option<VertexView<M, G>>, ()> {
        let perimeter = self.edges()
            .map(|edge| (edge.vertex, edge.next_edge().unwrap().vertex))
            .collect::<Vec<_>>();
        if perimeter.len() <= 3 {
            return Ok(None);
        }
        let face = self.geometry.clone();
        let centroid = G::centroid(self.with_mesh_ref())?;
        let mut mesh = self.remove()?;
        let c = mesh.as_mut().insert_vertex(centroid);
        for (a, b) in perimeter {
            let ab = mesh.as_mut()
                .insert_edge((a, b), G::Edge::default())
                .unwrap();
            let bc = mesh.as_mut()
                .insert_edge((b, c), G::Edge::default())
                .unwrap();
            let ca = mesh.as_mut()
                .insert_edge((c, a), G::Edge::default())
                .unwrap();
            mesh.as_mut()
                .insert_face(&[ab, bc, ca], face.clone())
                .unwrap();
        }
        Ok(Some(VertexView::new(mesh, c)))
    }
}

impl<M, G> FaceView<M, G>
where
    M: AsRef<Mesh<G>> + AsMut<Mesh<G>>,
    G: FaceNormal + Geometry,
    G::Vertex: AsPosition,
{
    pub fn extrude<T>(self, distance: T) -> Result<Self, ()>
    where
        G::Normal: Mul<T>,
        ScaledFaceNormal<G, T>: Clone,
        VertexPosition<G>: Add<ScaledFaceNormal<G, T>, Output = VertexPosition<G>> + Clone,
    {
        // Collect all the vertex keys of the face along with their translated
        // geometries.
        let vertices = self.extrude_vertex_geometry(distance)?;
        let face = self.geometry.clone();
        let mut mesh = self.remove()?;
        // Use the keys for the existing vertices and the translated geometries
        // to construct the extruded face and its connective faces.
        //
        // The winding of the faces is important; the extruded face must use
        // opposing edges to its neighboring connective faces.
        let extrusion = {
            let mesh = mesh.as_mut();
            let vertices = vertices
                .into_iter()
                .map(|vertex| (vertex.0, mesh.insert_vertex(vertex.1)))
                .collect::<Vec<_>>();
            let edges = vertices
                .duplet_circuit_windows()
                .map(|((_, a), (_, b))| {
                    mesh.insert_edge((a, b), G::Edge::default()).unwrap()
                })
                .collect::<Vec<_>>();
            let extrusion = mesh.insert_face(&edges, face).unwrap();
            for ((d, c), (a, b)) in vertices.duplet_circuit_windows() {
                let ab = mesh.insert_edge((a, b), G::Edge::default()).unwrap();
                let bc = mesh.insert_edge((b, c), G::Edge::default()).unwrap();
                let cd = mesh.insert_edge((c, d), G::Edge::default()).unwrap();
                let da = mesh.insert_edge((d, a), G::Edge::default()).unwrap();
                let ca = mesh.insert_edge((c, a), G::Edge::default()).unwrap(); // Diagonal.
                let ac = mesh.insert_edge((a, c), G::Edge::default()).unwrap(); // Diagonal.
                mesh.insert_face(&[ab, bc, ca], G::Face::default()).unwrap();
                mesh.insert_face(&[ac, cd, da], G::Face::default()).unwrap();
            }
            extrusion
        };
        Ok(FaceView::new(mesh, extrusion))
    }

    fn extrude_vertex_geometry<T>(&self, distance: T) -> Result<Vec<(VertexKey, G::Vertex)>, ()>
    where
        G::Normal: Mul<T>,
        ScaledFaceNormal<G, T>: Clone,
        VertexPosition<G>: Add<ScaledFaceNormal<G, T>, Output = VertexPosition<G>> + Clone,
    {
        let translation = G::normal(self.with_mesh_ref())? * distance;
        Ok(
            self.vertices()
                .map(|vertex| {
                    let mut geometry = vertex.geometry.clone();
                    *geometry.as_position_mut() =
                        geometry.as_position().clone() + translation.clone();
                    (vertex.key(), geometry)
                })
                .collect(),
        )
    }
}

impl<M, G> AsRef<FaceView<M, G>> for FaceView<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    fn as_ref(&self) -> &FaceView<M, G> {
        self
    }
}

impl<M, G> AsMut<FaceView<M, G>> for FaceView<M, G>
where
    M: AsRef<Mesh<G>> + AsMut<Mesh<G>>,
    G: Geometry,
{
    fn as_mut(&mut self) -> &mut FaceView<M, G> {
        self
    }
}

impl<M, G> Deref for FaceView<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    type Target = <Self as View<M, G>>::Topology;

    fn deref(&self) -> &Self::Target {
        self.mesh.as_ref().faces.get(&self.key).unwrap()
    }
}

impl<M, G> DerefMut for FaceView<M, G>
where
    M: AsRef<Mesh<G>> + AsMut<Mesh<G>>,
    G: Geometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.mesh.as_mut().faces.get_mut(&self.key).unwrap()
    }
}

impl<M, G> Clone for FaceView<M, G>
where
    M: AsRef<Mesh<G>> + Clone,
    G: Geometry,
{
    fn clone(&self) -> Self {
        FaceView {
            mesh: self.mesh.clone(),
            key: self.key.clone(),
            phantom: PhantomData,
        }
    }
}

impl<M, G> Copy for FaceView<M, G>
where
    M: AsRef<Mesh<G>> + Copy,
    G: Geometry,
{
}

impl<M, G> View<M, G> for FaceView<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    type Topology = Face<G>;

    fn from_mesh(mesh: M, key: <Self::Topology as Topological>::Key) -> Self {
        FaceView::new(mesh, key)
    }
}

pub struct OrphanFaceView<'a, G>
where
    G: 'a + Geometry,
{
    key: FaceKey,
    face: &'a mut Face<G>,
}

impl<'a, G> OrphanFaceView<'a, G>
where
    G: 'a + Geometry,
{
    pub(crate) fn new(face: &'a mut Face<G>, key: FaceKey) -> Self {
        OrphanFaceView {
            key: key,
            face: face,
        }
    }

    pub fn key(&self) -> FaceKey {
        self.key
    }
}

impl<'a, G> Deref for OrphanFaceView<'a, G>
where
    G: 'a + Geometry,
{
    type Target = <Self as OrphanView<'a, G>>::Topology;

    fn deref(&self) -> &Self::Target {
        &*self.face
    }
}

impl<'a, G> DerefMut for OrphanFaceView<'a, G>
where
    G: 'a + Geometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.face
    }
}

impl<'a, G> OrphanView<'a, G> for OrphanFaceView<'a, G>
where
    G: 'a + Geometry,
{
    type Topology = Face<G>;

    fn from_topology(
        topology: &'a mut Self::Topology,
        key: <Self::Topology as Topological>::Key,
    ) -> Self {
        OrphanFaceView::new(topology, key)
    }
}

pub struct FaceKeyTopology {
    key: FaceKey,
    edges: Vec<EdgeKeyTopology>,
}

impl FaceKeyTopology {
    fn new<I>(face: FaceKey, edges: I) -> Self
    where
        I: IntoIterator<Item = EdgeKeyTopology>,
    {
        FaceKeyTopology {
            key: face,
            edges: edges.into_iter().collect(),
        }
    }

    pub fn key(&self) -> FaceKey {
        self.key
    }

    pub fn edges(&self) -> &[EdgeKeyTopology] {
        self.edges.as_slice()
    }
}

pub struct VertexCirculator<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    inner: EdgeCirculator<M, G>,
}

impl<M, G> VertexCirculator<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    fn from_edge_circulator(edges: EdgeCirculator<M, G>) -> Self {
        VertexCirculator { inner: edges }
    }
}

impl<'a, G> Iterator for VertexCirculator<&'a Mesh<G>, G>
where
    G: Geometry,
{
    type Item = VertexView<&'a Mesh<G>, G>;

    fn next(&mut self) -> Option<Self::Item> {
        <EdgeCirculator<_, G> as Iterator>::next(&mut self.inner)
            .map(|edge| VertexView::new(self.inner.face.mesh, edge.vertex))
    }
}

impl<'a, G> Iterator for VertexCirculator<&'a mut Mesh<G>, G>
where
    G: Geometry,
{
    type Item = OrphanVertexView<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        <EdgeCirculator<_, G> as Iterator>::next(&mut self.inner)
            .map(|edge| {
                self.inner.face.mesh.edges.get(&edge.key()).unwrap().vertex
            })
            .map(|vertex| {
                OrphanVertexView::new(
                    unsafe {
                        use std::mem;

                        // There is no way to bind the anonymous lifetime of this
                        // function to `Self::Item`. This is problematic for the
                        // call to `get_mut`, which requires autoref. However, this
                        // should be safe, because the use of this iterator
                        // requires a mutable borrow of the source mesh with
                        // lifetime `'a`. Therefore, the (disjoint) geometry data
                        // within the mesh should also be valid over the lifetime
                        // '`a'.
                        mem::transmute::<_, &'a mut Vertex<G>>(
                            self.inner.face.mesh.vertices.get_mut(&vertex).unwrap(),
                        )
                    },
                    vertex,
                )
            })
    }
}

pub struct EdgeCirculator<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    face: FaceView<M, G>,
    edge: Option<EdgeKey>,
    breadcrumb: Option<EdgeKey>,
}

impl<M, G> EdgeCirculator<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    fn new(face: FaceView<M, G>) -> Self {
        let edge = face.edge;
        EdgeCirculator {
            face: face,
            edge: Some(edge),
            breadcrumb: Some(edge),
        }
    }

    fn next(&mut self) -> Option<EdgeKey> {
        self.edge.and_then(|edge| {
            let next = self.face.mesh.as_ref().edges.get(&edge).unwrap().next;
            self.breadcrumb.map(|_| {
                if self.breadcrumb == next {
                    self.breadcrumb = None;
                }
                else {
                    self.edge = next;
                }
                edge
            })
        })
    }
}

impl<'a, G> Iterator for EdgeCirculator<&'a Mesh<G>, G>
where
    G: Geometry,
{
    type Item = EdgeView<&'a Mesh<G>, G>;

    fn next(&mut self) -> Option<Self::Item> {
        <EdgeCirculator<_, _>>::next(self).map(|edge| EdgeView::new(self.face.mesh, edge))
    }
}

impl<'a, G> Iterator for EdgeCirculator<&'a mut Mesh<G>, G>
where
    G: Geometry,
{
    type Item = OrphanEdgeView<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        <EdgeCirculator<_, _>>::next(self).map(|edge| {
            OrphanEdgeView::new(
                unsafe {
                    use std::mem;

                    // There is no way to bind the anonymous lifetime of this
                    // function to `Self::Item`. This is problematic for the
                    // call to `get_mut`, which requires autoref. However, this
                    // should be safe, because the use of this iterator
                    // requires a mutable borrow of the source mesh with
                    // lifetime `'a`. Therefore, the (disjoint) geometry data
                    // within the mesh should also be valid over the lifetime
                    // '`a'.
                    mem::transmute::<_, &'a mut Edge<G>>(
                        self.face.mesh.edges.get_mut(&edge).unwrap(),
                    )
                },
                edge,
            )
        })
    }
}

pub struct FaceCirculator<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    inner: EdgeCirculator<M, G>,
}

impl<M, G> FaceCirculator<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    fn from_edge_circulator(edges: EdgeCirculator<M, G>) -> Self {
        FaceCirculator { inner: edges }
    }

    fn next(&mut self) -> Option<FaceKey> {
        while let Some(edge) = self.inner.next().map(|edge| {
            self.inner.face.mesh.as_ref().edges.get(&edge).unwrap()
        }) {
            if let Some(face) = edge.opposite
                .map(|opposite| {
                    self.inner.face.mesh.as_ref().edges.get(&opposite).unwrap()
                })
                .and_then(|opposite| opposite.face)
            {
                return Some(face);
            }
            else {
                // Skip edges with no opposing face. This can occur within
                // non-enclosed meshes.
                continue;
            }
        }
        None
    }
}

impl<'a, G> Iterator for FaceCirculator<&'a Mesh<G>, G>
where
    G: Geometry,
{
    type Item = FaceView<&'a Mesh<G>, G>;

    fn next(&mut self) -> Option<Self::Item> {
        <FaceCirculator<_, _>>::next(self).map(|face| FaceView::new(self.inner.face.mesh, face))
    }
}

impl<'a, G> Iterator for FaceCirculator<&'a mut Mesh<G>, G>
where
    G: 'a + Geometry,
{
    // This cannot be a `FaceView`, because that would alias the mutable
    // reference to the mesh. Instead, yield the key and a mutable reference to
    // the geometry data as an `OrphanFaceView` that discards any traversable
    // reference into the mesh.
    type Item = OrphanFaceView<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        <FaceCirculator<_, _>>::next(self).map(|face| {
            OrphanFaceView::new(
                unsafe {
                    use std::mem;

                    // There is no way to bind the anonymous lifetime of this
                    // function to `Self::Item`. This is problematic for the
                    // call to `get_mut`, which requires autoref. However, this
                    // should be safe, because the use of this iterator
                    // requires a mutable borrow of the source mesh with
                    // lifetime `'a`. Therefore, the (disjoint) geometry data
                    // within the mesh should also be valid over the lifetime
                    // '`a'.
                    mem::transmute::<_, &'a mut Face<G>>(
                        self.inner.face.mesh.faces.get_mut(&face).unwrap(),
                    )
                },
                face,
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use decorum::R32;
    use nalgebra::Point3;

    use generate::*;
    use graph::*;

    #[test]
    fn circulate_over_edges() {
        let mesh = sphere::UVSphere::<R32>::with_unit_radius(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<Mesh<Point3<f32>>>();
        let face = mesh.faces().nth(0).unwrap();

        // All faces should be triangles and should have three edges.
        assert_eq!(3, face.edges().count());
    }

    #[test]
    fn circulate_over_faces() {
        let mesh = sphere::UVSphere::<R32>::with_unit_radius(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<Mesh<Point3<f32>>>();
        let face = mesh.faces().nth(0).unwrap();

        // No matter which face is selected, it should have three neighbors.
        assert_eq!(3, face.faces().count());
    }

    #[test]
    fn extrude_face() {
        let mut mesh = sphere::UVSphere::<R32>::with_unit_radius(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<Mesh<Point3<f32>>>();
        {
            let key = mesh.faces().nth(0).unwrap().key();
            let face = mesh.face_mut(key).unwrap().extrude(1.0).unwrap();

            // The extruded face, being a triangle, should have three
            // neighboring faces.
            assert_eq!(3, face.faces().count());
        }

        assert_eq!(8, mesh.vertex_count());
        // The mesh begins with 18 edges. The additional edges are derived from
        // seven new triangles, but three edges are shared, so there are `(7 *
        // 3) - 3` new edges.
        assert_eq!(36, mesh.edge_count());
        // All faces are triangles and the mesh begins with six such faces. The
        // extruded face remains, in addition to three connective faces, each
        // of which is constructed from two triangular faces.
        assert_eq!(12, mesh.face_count());
    }

    #[test]
    fn triangulate_mesh() {
        let (indeces, vertices) = cube::Cube::<R32>::with_unit_radius()
            .polygons_with_position() // 6 quads, 24 vertices.
            .index_vertices(HashIndexer::default());
        let mut mesh = Mesh::<Point3<f32>>::from_raw_buffers(indeces, vertices, 4).unwrap();
        mesh.triangulate().unwrap();

        // There are 8 unique vertices and a vertex is added for each quad,
        // yielding `8 + 6` vertices.
        assert_eq!(14, mesh.vertex_count());
        assert_eq!(72, mesh.edge_count());
        // Each quad becomes a tetrahedron, so 6 quads become 24 triangles.
        assert_eq!(24, mesh.face_count());
    }
}
