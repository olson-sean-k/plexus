use std::marker::PhantomData;
use std::ops::{Add, Deref, DerefMut, Mul};

use graph::geometry::{AsPosition, FaceNormal, Geometry};
use graph::geometry::alias::{ScaledFaceNormal, VertexPosition};
use graph::mesh::{Edge, Face, Mesh, Vertex};
use graph::storage::{EdgeKey, FaceKey, VertexKey};
use graph::topology::{EdgeView, OrphanEdgeView, OrphanVertexView, VertexView};

#[derive(Clone, Copy)]
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

    // Resolve the `M` parameter to a concrete reference.
    fn with_mesh_mut(&mut self) -> FaceView<&mut Mesh<G>, G> {
        FaceView::new(self.mesh.as_mut(), self.key)
    }
}

// The elaborate type constraints state that positions stored in the vertex
// geometry can be used to compute a normal of a face and that normal can be
// scaled and added to vertex positions.
impl<M, G> FaceView<M, G>
where
    M: AsRef<Mesh<G>> + AsMut<Mesh<G>>,
    G: FaceNormal + Geometry,
    G::Vertex: AsPosition,
{
    pub fn extrude<F>(self, distance: F) -> Result<Self, ()>
    where
        G::Normal: Mul<F>,
        ScaledFaceNormal<G, F>: Clone,
        VertexPosition<G>: Add<ScaledFaceNormal<G, F>, Output = VertexPosition<G>> + Clone,
    {
        self.extrude_with_geometry(distance, G::Edge::default(), G::Face::default())
    }

    pub fn extrude_with_geometry<F>(
        self,
        distance: F,
        edge: G::Edge,
        face: G::Face,
    ) -> Result<Self, ()>
    where
        G::Normal: Mul<F>,
        ScaledFaceNormal<G, F>: Clone,
        VertexPosition<G>: Add<ScaledFaceNormal<G, F>, Output = VertexPosition<G>> + Clone,
    {
        // Collect all the vertex keys of the face along with their translated
        // geometries.
        let geometry = self.extrude_vertex_geometry(distance)?;
        // Begin topological mutations, starting by removing the originating
        // face. These mutations invalidate the key used by the `FaceView`, so
        // destructure `self` to avoid its reuse.
        let FaceView { mut mesh, key, .. } = self;
        mesh.as_mut().remove_face(key).unwrap();
        // Use the keys for the existing vertices and the translated geometries
        // to construct the extruded face and its connective faces.
        //
        // The winding of the faces is important; the extruded face must use
        // opposing edges to its neighboring connective faces.
        let extrusion = {
            let mesh = mesh.as_mut();
            let vertices = geometry
                .into_iter()
                .map(|vertex| (vertex.0, mesh.insert_vertex(vertex.1)))
                .collect::<Vec<_>>();
            let edges = vertices
                .iter()
                .enumerate()
                .map(|(index, &(_, a))| {
                    let b = vertices[(index + 1) % vertices.len()].1;
                    mesh.insert_edge((a, b), edge.clone()).unwrap()
                })
                .collect::<Vec<_>>();
            let extrusion = mesh.insert_face(&edges, face.clone()).unwrap();
            for index in 0..vertices.len() {
                let (d, c) = vertices[index];
                let (a, b) = vertices[(index + 1) % vertices.len()];
                let ab = mesh.insert_edge((a, b), edge.clone()).unwrap();
                let bc = mesh.insert_edge((b, c), edge.clone()).unwrap();
                let cd = mesh.insert_edge((c, d), edge.clone()).unwrap();
                let da = mesh.insert_edge((d, a), edge.clone()).unwrap();
                let ca = mesh.insert_edge((c, a), edge.clone()).unwrap(); // Diagonal.
                let ac = mesh.insert_edge((a, c), edge.clone()).unwrap(); // Diagonal.
                mesh.insert_face(&[ab, bc, ca], face.clone()).unwrap();
                mesh.insert_face(&[ac, cd, da], face.clone()).unwrap();
            }
            extrusion
        };
        Ok(FaceView::new(mesh, extrusion))
    }

    fn extrude_vertex_geometry<F>(&self, distance: F) -> Result<Vec<(VertexKey, G::Vertex)>, ()>
    where
        // Constraints on the normal of the face:
        //
        // 1. Supports multiplication with `F` (becoming the scaled face
        //    normal).
        G::Normal: Mul<F>,
        // Constraints on the scaled normal of the face:
        //
        // 1. Supports cloning.
        ScaledFaceNormal<G, F>: Clone,
        // Constraints on the vertex geometry when converted to a position:
        //
        // 1. Supports cloning.
        // 2. Supports addition with the scaled normal of the face.
        // 3. The output of the above addition is itself.
        VertexPosition<G>: Add<ScaledFaceNormal<G, F>, Output = VertexPosition<G>> + Clone,
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
    type Target = Face<G>;

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

// There's no need to abstract over mutability for this type. For immutable
// refs, there is no need for an orphan type. Moreover, it is not possible to
// implement `AsRef` and `AsMut` for all types that implement `Geometry`.
pub struct OrphanFaceView<'a, G>
where
    G: 'a + Geometry,
{
    key: FaceKey,
    // The name `geometry` mirrors the `geometry` field of `Face`, to which
    // `FaceView` derefs.
    pub geometry: &'a mut G::Face,
}

impl<'a, G> OrphanFaceView<'a, G>
where
    G: 'a + Geometry,
{
    pub(crate) fn new(geometry: &'a mut G::Face, face: FaceKey) -> Self {
        OrphanFaceView {
            key: face,
            geometry: geometry,
        }
    }

    pub fn key(&self) -> FaceKey {
        self.key
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
                let geometry = {
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
                        let vertex = mem::transmute::<_, &'a mut Vertex<G>>(
                            self.inner.face.mesh.vertices.get_mut(&vertex).unwrap(),
                        );
                        &mut vertex.geometry
                    }
                };
                OrphanVertexView::new(geometry, vertex)
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
            let geometry = {
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
                    let edge = mem::transmute::<_, &'a mut Edge<G>>(
                        self.face.mesh.edges.get_mut(&edge).unwrap(),
                    );
                    &mut edge.geometry
                }
            };
            OrphanEdgeView::new(geometry, edge)
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
            let geometry = {
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
                    let face = mem::transmute::<_, &'a mut Face<G>>(
                        self.inner.face.mesh.faces.get_mut(&face).unwrap(),
                    );
                    &mut face.geometry
                }
            };
            OrphanFaceView::new(geometry, face)
        })
    }
}

#[cfg(test)]
mod tests {
    use generate::*;
    use graph::*;
    use ordered::*;

    #[test]
    fn circulate_over_edges() {
        let mesh = sphere::UVSphere::<f32>::with_unit_radius(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .map_vertices(|vertex| vertex.into_hash())
            .collect::<Mesh<Triplet<_>>>();
        // TODO: Provide a way to get a key for the faces in the mesh. Using
        //       `default` only works if the initial face has not been removed.
        let face = mesh.face(FaceKey::default()).unwrap();

        // All faces should be triangles and should have three edges.
        assert_eq!(3, face.edges().count());
    }

    #[test]
    fn circulate_over_faces() {
        let mesh = sphere::UVSphere::<f32>::with_unit_radius(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .map_vertices(|vertex| vertex.into_hash())
            .collect::<Mesh<Triplet<_>>>();
        // TODO: Provide a way to get a key for the faces in the mesh. Using
        //       `default` only works if the initial face has not been removed.
        let face = mesh.face(FaceKey::default()).unwrap();

        // No matter which face is selected, it should have three neighbors.
        assert_eq!(3, face.faces().count());
    }

    #[test]
    fn extrude_face() {
        use nalgebra::Point3;

        let mut mesh = sphere::UVSphere::<f32>::with_unit_radius(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect_with_indexer::<Mesh<Point3<f32>>, _>(LruIndexer::default());
        {
            // TODO: Provide a way to get a key for the faces in the mesh.
            //       Using `default` only works if the initial face has not
            //       been removed.
            let face = mesh.face_mut(FaceKey::default()).unwrap();
            let face = face.extrude(1.0).unwrap();

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
}
