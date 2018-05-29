use failure::Error;
use std::marker::PhantomData;
use std::ops::{Add, Deref, DerefMut, Mul};

use geometry::convert::AsPosition;
use geometry::Geometry;
use graph::geometry::alias::{ScaledFaceNormal, VertexPosition};
use graph::geometry::{FaceCentroid, FaceNormal};
use graph::mesh::{Edge, Face, Mesh, Vertex};
use graph::mutation::{ModalMutation, Mutation};
use graph::storage::{EdgeKey, FaceKey, VertexKey};
use graph::topology::{
    edge, EdgeKeyTopology, EdgeView, OrphanEdgeView, OrphanVertexView, OrphanView, Topological,
    VertexView, View,
};
use graph::{GraphError, Perimeter};

/// Do **not** use this type directly. Use `FaceRef` and `FaceMut` instead.
///
/// This type is only re-exported so that its members are shown in
/// documentation. See this issue:
/// <https://github.com/rust-lang/rust/issues/39437>
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
    pub(in graph) fn new(mesh: M, face: FaceKey) -> Self {
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

    // Resolve the `M` parameter to a concrete reference.
    fn with_mesh_mut(&mut self) -> FaceView<&mut Mesh<G>, G> {
        FaceView::new(self.mesh.as_mut(), self.key)
    }
}

impl<'a, G> FaceView<&'a mut Mesh<G>, G>
where
    G: Geometry,
{
    pub fn join(self, destination: FaceKey) -> Result<(), Error> {
        let FaceView {
            mesh, key: source, ..
        } = self;
        let mut mutation = Mutation::replace(mesh, Mesh::empty());
        join(&mut *mutation, source, destination)?;
        mutation.commit()?;
        Ok(())
    }
}

impl<M, G> FaceView<M, G>
where
    M: AsRef<Mesh<G>>,
    G: FaceCentroid + Geometry,
{
    pub fn centroid(&self) -> Result<G::Centroid, Error> {
        G::centroid(self.with_mesh_ref())
    }
}

impl<'a, G> FaceView<&'a mut Mesh<G>, G>
where
    G: FaceCentroid<Centroid = <G as Geometry>::Vertex> + Geometry,
{
    pub fn triangulate(self) -> Result<Option<VertexView<&'a mut Mesh<G>, G>>, Error> {
        let FaceView { mesh, key: abc, .. } = self;
        let mut mutation = Mutation::immediate(mesh);
        Ok(match triangulate(&mut mutation, abc)? {
            Some(vertex) => Some(VertexView::new(mutation.commit(), vertex)),
            _ => None,
        })
    }
}

impl<M, G> FaceView<M, G>
where
    M: AsRef<Mesh<G>>,
    G: FaceNormal + Geometry,
{
    pub fn normal(&self) -> Result<G::Normal, Error> {
        G::normal(self.with_mesh_ref())
    }
}

impl<'a, G> FaceView<&'a mut Mesh<G>, G>
where
    G: FaceNormal + Geometry,
    G::Vertex: AsPosition,
{
    pub fn extrude<T>(self, distance: T) -> Result<FaceView<&'a mut Mesh<G>, G>, Error>
    where
        G::Normal: Mul<T>,
        ScaledFaceNormal<G, T>: Clone,
        VertexPosition<G>: Add<ScaledFaceNormal<G, T>, Output = VertexPosition<G>> + Clone,
    {
        let FaceView { mesh, key: abc, .. } = self;
        let mut mutation = Mutation::replace(mesh, Mesh::empty());
        let face = extrude(&mut *mutation, abc, distance)?;
        Ok(FaceView::new(mutation.commit().unwrap(), face))
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
            key: self.key,
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

/// Do **not** use this type directly. Use `OrphanFaceMut` instead.
///
/// This type is only re-exported so that its members are shown in
/// documentation. See this issue:
/// <https://github.com/rust-lang/rust/issues/39437>
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
    pub(in graph) fn new(face: &'a mut Face<G>, key: FaceKey) -> Self {
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

#[derive(Clone, Debug)]
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
            .map(|edge| self.inner.face.mesh.edges.get(&edge.key()).unwrap().vertex)
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
        while let Some(edge) = self
            .inner
            .next()
            .map(|edge| self.inner.face.mesh.as_ref().edges.get(&edge).unwrap())
        {
            if let Some(face) = edge
                .opposite
                .map(|opposite| self.inner.face.mesh.as_ref().edges.get(&opposite).unwrap())
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

pub(in graph) fn triangulate<'a, M, G>(
    mutation: &mut M,
    abc: FaceKey,
) -> Result<Option<VertexKey>, Error>
where
    M: ModalMutation<'a, G>,
    G: 'a + FaceCentroid<Centroid = <G as Geometry>::Vertex> + Geometry,
{
    let (perimeter, centroid, face) = {
        let face = match mutation.as_mesh().face(abc) {
            Some(face) => face,
            _ => return Err(GraphError::TopologyNotFound.into()),
        };
        let perimeter = face
            .edges()
            .map(|edge| (edge.vertex, edge.next_edge().vertex))
            .collect::<Vec<_>>();
        if perimeter.len() <= 3 {
            return Ok(None);
        }
        (perimeter, face.centroid()?, face.geometry.clone())
    };
    // This is the point of no return; the mesh has mutated. Unwrap
    // results.
    mutation.remove_face(abc).unwrap();
    let c = mutation.insert_vertex(centroid);
    for (a, b) in perimeter {
        mutation
            .insert_face(&[a, b, c], (Default::default(), face.clone()))
            .unwrap();
    }
    Ok(Some(c))
}

pub(in graph) fn join<'a, M, G>(
    mutation: &mut M,
    source: FaceKey,
    destination: FaceKey,
) -> Result<(), Error>
where
    M: ModalMutation<'a, G>,
    G: 'a + Geometry,
{
    let (sources, destination) = {
        let source = match mutation.as_mesh().face(source) {
            Some(face) => face,
            _ => return Err(GraphError::TopologyNotFound.into()),
        };
        // Ensure that the opposite face exists and has the same arity.
        let arity = source.arity();
        let destination = match mutation.as_mesh().face(destination) {
            Some(destination) => {
                if destination.arity() != arity {
                    return Err(GraphError::ArityNonConstant.into());
                }
                destination
            }
            _ => {
                return Err(GraphError::TopologyNotFound.into());
            }
        };
        // Decompose the faces into their key topologies.
        (
            source
                .to_key_topology()
                .edges()
                .iter()
                .map(|topology| {
                    (
                        topology.clone(),
                        mutation
                            .as_mesh()
                            .edge(topology.key())
                            .unwrap()
                            .geometry
                            .clone(),
                    )
                })
                .collect::<Vec<_>>(),
            destination.to_key_topology(),
        )
    };
    // Remove the source and destination faces. Pair the topology with edge
    // geometry for the source face. At this point, we can assume that the
    // faces exist and no failures should occur; unwrap results.
    mutation.remove_face(source)?;
    mutation.remove_face(destination.key())?;
    // TODO: Is it always correct to reverse the order of the opposite
    //       face's edges?
    // Re-insert the edges of the faces and join the mutual edges.
    for (source, destination) in sources.into_iter().zip(destination.edges().iter().rev()) {
        let (a, b) = source.0.vertices();
        let (c, d) = destination.vertices();
        let ab = mutation.insert_edge((a, b), source.1.clone()).unwrap();
        let cd = mutation.insert_edge((c, d), source.1).unwrap();
        edge::join(mutation, ab, cd).unwrap();
    }
    // TODO: Is there any reasonable topology this can return?
    Ok(())
}

pub(in graph) fn extrude<'a, M, G, T>(
    mutation: &mut M,
    abc: FaceKey,
    distance: T,
) -> Result<FaceKey, Error>
where
    M: ModalMutation<'a, G>,
    G: 'a + FaceNormal + Geometry,
    G::Normal: Mul<T>,
    G::Vertex: AsPosition,
    ScaledFaceNormal<G, T>: Clone,
    VertexPosition<G>: Add<ScaledFaceNormal<G, T>, Output = VertexPosition<G>> + Clone,
{
    // Collect all the vertex keys of the face along with their translated
    // geometries.
    let (sources, destinations, face) = {
        let face = match mutation.as_mesh().face(abc) {
            Some(face) => face,
            _ => return Err(GraphError::TopologyNotFound.into()),
        };
        let translation = face.normal()? * distance;
        let sources = face
            .vertices()
            .map(|vertex| vertex.key())
            .collect::<Vec<_>>();
        let destinations = face
            .vertices()
            .map(|vertex| {
                let mut geometry = vertex.geometry.clone();
                *geometry.as_position_mut() = geometry.as_position().clone() + translation.clone();
                geometry
            })
            .collect::<Vec<_>>();
        (sources, destinations, face.geometry.clone())
    };
    // This is the point of no return; the mesh has been mutated. Unwrap
    // results.
    mutation.remove_face(abc).unwrap();
    let destinations = destinations
        .into_iter()
        .map(|vertex| mutation.insert_vertex(vertex))
        .collect::<Vec<_>>();
    // Use the keys for the existing vertices and the translated geometries
    // to construct the extruded face and its connective faces.
    let extrusion = mutation
        .insert_face(&destinations, (Default::default(), face))
        .unwrap();
    for ((a, c), (b, d)) in sources
        .into_iter()
        .zip(destinations.into_iter())
        .collect::<Vec<_>>()
        .perimeter()
    {
        // TODO: Split these faces to form triangles.
        mutation
            .insert_face(&[a, b, d, c], Default::default())
            .unwrap();
    }
    Ok(extrusion)
}

#[cfg(test)]
mod tests {
    use nalgebra::Point3;

    use generate::*;
    use graph::*;

    #[test]
    fn circulate_over_edges() {
        let mesh = sphere::UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<Mesh<Point3<f32>>>();
        let face = mesh.faces().nth(0).unwrap();

        // All faces should be triangles and should have three edges.
        assert_eq!(3, face.edges().count());
    }

    #[test]
    fn circulate_over_faces() {
        let mesh = sphere::UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<Mesh<Point3<f32>>>();
        let face = mesh.faces().nth(0).unwrap();

        // No matter which face is selected, it should have three neighbors.
        assert_eq!(3, face.faces().count());
    }

    #[test]
    fn extrude_face() {
        let mut mesh = sphere::UvSphere::new(3, 2)
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
        // The mesh begins with 18 edges. The extrusion adds three quads with
        // four interior edges each, so there are `18 + (3 * 4)` edges.
        assert_eq!(30, mesh.edge_count());
        // All faces are triangles and the mesh begins with six such faces. The
        // extruded face remains, in addition to three connective faces, each
        // of which is constructed from quads.
        assert_eq!(9, mesh.face_count());
    }

    #[test]
    fn triangulate_mesh() {
        let (indeces, vertices) = cube::Cube::new()
            .polygons_with_position() // 6 quads, 24 vertices.
            .flat_index_vertices(HashIndexer::default());
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
