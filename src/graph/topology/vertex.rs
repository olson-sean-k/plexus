use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use geometry::Geometry;
use graph::mesh::{Edge, Face, Mesh, Vertex};
use graph::storage::{EdgeKey, FaceKey, VertexKey};
use graph::topology::{EdgeView, FaceView, OrphanEdgeView, OrphanFaceView, OrphanView, Topological,
                      View};

/// Do **not** use this type directly. Use `VertexRef` and `VertexMut` instead.
///
/// This type is only re-exported so that its members are shown in
/// documentation. See this issue:
/// <https://github.com/rust-lang/rust/issues/39437>
pub struct VertexView<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    mesh: M,
    key: VertexKey,
    phantom: PhantomData<G>,
}

impl<M, G> VertexView<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    pub(in graph) fn new(mesh: M, vertex: VertexKey) -> Self {
        VertexView {
            mesh: mesh,
            key: vertex,
            phantom: PhantomData,
        }
    }

    pub fn key(&self) -> VertexKey {
        self.key
    }

    pub fn outgoing_edge(&self) -> EdgeView<&Mesh<G>, G> {
        self.raw_outgoing_edge().unwrap()
    }

    pub fn into_outgoing_edge(self) -> EdgeView<M, G> {
        self.into_raw_outgoing_edge().unwrap()
    }

    pub fn incoming_edges(&self) -> EdgeCirculator<&Mesh<G>, G> {
        EdgeCirculator::new(self.with_mesh_ref())
    }

    pub fn faces(&self) -> FaceCirculator<&Mesh<G>, G> {
        FaceCirculator::from_edge_circulator(self.incoming_edges())
    }

    // Resolve the `M` parameter to a concrete reference.
    fn with_mesh_ref(&self) -> VertexView<&Mesh<G>, G> {
        VertexView::new(self.mesh.as_ref(), self.key)
    }
}

/// Raw API.
impl<M, G> VertexView<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    pub(in graph) fn raw_outgoing_edge(&self) -> Option<EdgeView<&Mesh<G>, G>> {
        self.edge
            .map(|edge| EdgeView::new(self.mesh.as_ref(), edge))
    }

    pub(in graph) fn into_raw_outgoing_edge(self) -> Option<EdgeView<M, G>> {
        let edge = self.edge;
        let mesh = self.mesh;
        edge.map(|edge| EdgeView::new(mesh, edge))
    }
}

impl<M, G> VertexView<M, G>
where
    M: AsRef<Mesh<G>> + AsMut<Mesh<G>>,
    G: Geometry,
{
    pub fn outgoing_edge_mut(&mut self) -> OrphanEdgeView<G> {
        self.raw_outgoing_edge_mut().unwrap()
    }

    pub fn incoming_edges_mut(&mut self) -> EdgeCirculator<&mut Mesh<G>, G> {
        EdgeCirculator::new(self.with_mesh_mut())
    }

    pub fn faces_mut(&mut self) -> FaceCirculator<&mut Mesh<G>, G> {
        FaceCirculator::from_edge_circulator(self.incoming_edges_mut())
    }

    // Resolve the `M` parameter to a concrete reference.
    fn with_mesh_mut(&mut self) -> VertexView<&mut Mesh<G>, G> {
        VertexView::new(self.mesh.as_mut(), self.key)
    }
}

/// Raw API.
impl<M, G> VertexView<M, G>
where
    M: AsRef<Mesh<G>> + AsMut<Mesh<G>>,
    G: Geometry,
{
    pub(in graph) fn raw_outgoing_edge_mut(&mut self) -> Option<OrphanEdgeView<G>> {
        let edge = self.edge;
        edge.map(move |edge| self.mesh.as_mut().orphan_edge_mut(edge).unwrap())
    }
}

impl<M, G> AsRef<VertexView<M, G>> for VertexView<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    fn as_ref(&self) -> &VertexView<M, G> {
        self
    }
}

impl<M, G> AsMut<VertexView<M, G>> for VertexView<M, G>
where
    M: AsRef<Mesh<G>> + AsMut<Mesh<G>>,
    G: Geometry,
{
    fn as_mut(&mut self) -> &mut VertexView<M, G> {
        self
    }
}

impl<M, G> Deref for VertexView<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    type Target = Vertex<G>;

    fn deref(&self) -> &Self::Target {
        self.mesh.as_ref().vertices.get(&self.key).unwrap()
    }
}

impl<M, G> DerefMut for VertexView<M, G>
where
    M: AsRef<Mesh<G>> + AsMut<Mesh<G>>,
    G: Geometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.mesh.as_mut().vertices.get_mut(&self.key).unwrap()
    }
}

impl<M, G> Clone for VertexView<M, G>
where
    M: AsRef<Mesh<G>> + Clone,
    G: Geometry,
{
    fn clone(&self) -> Self {
        VertexView {
            mesh: self.mesh.clone(),
            key: self.key,
            phantom: PhantomData,
        }
    }
}

impl<M, G> Copy for VertexView<M, G>
where
    M: AsRef<Mesh<G>> + Copy,
    G: Geometry,
{
}

impl<M, G> View<M, G> for VertexView<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    type Topology = Vertex<G>;

    fn from_mesh(mesh: M, key: <Self::Topology as Topological>::Key) -> Self {
        VertexView::new(mesh, key)
    }
}

/// Do **not** use this type directly. Use `OrphanVertexMut` instead.
///
/// This type is only re-exported so that its members are shown in
/// documentation. See this issue:
/// <https://github.com/rust-lang/rust/issues/39437>
pub struct OrphanVertexView<'a, G>
where
    G: 'a + Geometry,
{
    key: VertexKey,
    vertex: &'a mut Vertex<G>,
}

impl<'a, G> OrphanVertexView<'a, G>
where
    G: 'a + Geometry,
{
    pub(in graph) fn new(vertex: &'a mut Vertex<G>, key: VertexKey) -> Self {
        OrphanVertexView {
            key: key,
            vertex: vertex,
        }
    }

    pub fn key(&self) -> VertexKey {
        self.key
    }
}

impl<'a, G> Deref for OrphanVertexView<'a, G>
where
    G: 'a + Geometry,
{
    type Target = <Self as OrphanView<'a, G>>::Topology;

    fn deref(&self) -> &Self::Target {
        &*self.vertex
    }
}

impl<'a, G> DerefMut for OrphanVertexView<'a, G>
where
    G: 'a + Geometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.vertex
    }
}

impl<'a, G> OrphanView<'a, G> for OrphanVertexView<'a, G>
where
    G: 'a + Geometry,
{
    type Topology = Vertex<G>;

    fn from_topology(
        topology: &'a mut Self::Topology,
        key: <Self::Topology as Topological>::Key,
    ) -> Self {
        OrphanVertexView::new(topology, key)
    }
}

pub struct EdgeCirculator<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    vertex: VertexView<M, G>,
    edge: Option<EdgeKey>,
    breadcrumb: Option<EdgeKey>,
}

impl<M, G> EdgeCirculator<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    fn new(vertex: VertexView<M, G>) -> Self {
        let edge = vertex.edge;
        EdgeCirculator {
            vertex: vertex,
            edge: edge,
            breadcrumb: edge,
        }
    }

    fn next(&mut self) -> Option<EdgeKey> {
        self.edge
            .map(|outgoing| self.vertex.mesh.as_ref().edges.get(&outgoing).unwrap())
            .and_then(|outgoing| outgoing.opposite)
            .and_then(|incoming| {
                let outgoing = self.vertex.mesh.as_ref().edges.get(&incoming).unwrap().next;
                self.breadcrumb.map(|_| {
                    if self.breadcrumb == outgoing {
                        self.breadcrumb = None;
                    }
                    else {
                        self.edge = outgoing;
                    }
                    incoming
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
        <EdgeCirculator<_, _>>::next(self).map(|edge| EdgeView::new(self.vertex.mesh, edge))
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
                        self.vertex.mesh.edges.get_mut(&edge).unwrap(),
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
        while let Some(edge) = self.inner
            .next()
            .map(|edge| self.inner.vertex.mesh.as_ref().edges.get(&edge).unwrap())
        {
            if let Some(face) = edge.face {
                return Some(face);
            }
            else {
                // Skip edges with no face. This can occur within non-enclosed
                // meshes.
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
        <FaceCirculator<_, _>>::next(self).map(|face| FaceView::new(self.inner.vertex.mesh, face))
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
                        self.inner.vertex.mesh.faces.get_mut(&face).unwrap(),
                    )
                },
                face,
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Point3;

    use generate::*;
    use graph::*;

    #[test]
    fn circulate_over_edges() {
        let mesh = sphere::UvSphere::new(4, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<Mesh<Point3<f32>>>();

        // All faces should be triangles and all vertices should have 4
        // (incoming) edges.
        for vertex in mesh.vertices() {
            assert_eq!(4, vertex.incoming_edges().count());
        }
    }
}
