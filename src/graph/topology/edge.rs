use failure::Error;
use std::marker::PhantomData;
use std::ops::{Add, Deref, DerefMut, Mul};

use geometry::Geometry;
use geometry::convert::AsPosition;
use graph::{GraphError, Perimeter};
use graph::geometry::{EdgeLateral, EdgeMidpoint};
use graph::geometry::alias::{ScaledEdgeLateral, VertexPosition};
use graph::mesh::{Edge, Mesh};
use graph::mutation::Mutation;
use graph::storage::{EdgeKey, VertexKey};
use graph::topology::{FaceView, OrphanFaceView, OrphanVertexView, OrphanView, Topological,
                      VertexView, View};

/// Do **not** use this type directly. Use `EdgeRef` and `EdgeMut` instead.
///
/// This type is only re-exported so that its members are shown in
/// documentation. See this issue:
/// <https://github.com/rust-lang/rust/issues/39437>
pub struct EdgeView<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    mesh: M,
    key: EdgeKey,
    phantom: PhantomData<G>,
}

impl<M, G> EdgeView<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    pub(in graph) fn new(mesh: M, edge: EdgeKey) -> Self {
        EdgeView {
            mesh: mesh,
            key: edge,
            phantom: PhantomData,
        }
    }

    pub fn key(&self) -> EdgeKey {
        self.key
    }

    pub fn to_key_topology(&self) -> EdgeKeyTopology {
        EdgeKeyTopology::new(self.key, self.key.to_vertex_keys())
    }

    pub fn source_vertex(&self) -> VertexView<&Mesh<G>, G> {
        let (vertex, _) = self.key.to_vertex_keys();
        VertexView::new(self.mesh.as_ref(), vertex)
    }

    pub fn into_source_vertex(self) -> VertexView<M, G> {
        let (vertex, _) = self.key.to_vertex_keys();
        let mesh = self.mesh;
        VertexView::new(mesh, vertex)
    }

    pub fn destination_vertex(&self) -> VertexView<&Mesh<G>, G> {
        VertexView::new(self.mesh.as_ref(), self.vertex)
    }

    pub fn into_destination_vertex(self) -> VertexView<M, G> {
        let vertex = self.vertex;
        let mesh = self.mesh;
        VertexView::new(mesh, vertex)
    }

    pub fn opposite_edge(&self) -> EdgeView<&Mesh<G>, G> {
        self.raw_opposite_edge().unwrap()
    }

    pub fn into_opposite_edge(self) -> Self {
        self.into_raw_opposite_edge().unwrap()
    }

    pub fn next_edge(&self) -> EdgeView<&Mesh<G>, G> {
        self.raw_next_edge().unwrap()
    }

    pub fn into_next_edge(self) -> Self {
        self.into_raw_next_edge().unwrap()
    }

    pub fn previous_edge(&self) -> EdgeView<&Mesh<G>, G> {
        self.raw_previous_edge().unwrap()
    }

    pub fn into_previous_edge(self) -> Self {
        self.into_raw_previous_edge().unwrap()
    }

    pub fn face(&self) -> Option<FaceView<&Mesh<G>, G>> {
        self.face
            .map(|face| FaceView::new(self.mesh.as_ref(), face))
    }

    pub fn into_face(self) -> Option<FaceView<M, G>> {
        let face = self.face;
        let mesh = self.mesh;
        face.map(|face| FaceView::new(mesh, face))
    }

    pub fn boundary_edge(&self) -> Option<EdgeView<&Mesh<G>, G>> {
        use BoolExt;

        if self.is_boundary_edge() {
            Some(self.with_mesh_ref())
        }
        else {
            let opposite = self.opposite_edge();
            opposite.is_boundary_edge().into_some(opposite)
        }
    }

    pub fn into_boundary_edge(self) -> Option<Self> {
        use BoolExt;

        if self.is_boundary_edge() {
            Some(self)
        }
        else {
            let opposite = self.into_opposite_edge();
            opposite.is_boundary_edge().into_some(opposite)
        }
    }

    pub fn is_boundary_edge(&self) -> bool {
        self.face().is_none()
    }

    // Resolve the `M` parameter to a concrete reference.
    #[allow(dead_code)]
    fn with_mesh_ref(&self) -> EdgeView<&Mesh<G>, G> {
        EdgeView::new(self.mesh.as_ref(), self.key)
    }
}

/// Raw API.
impl<M, G> EdgeView<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    pub(in graph) fn raw_opposite_edge(&self) -> Option<EdgeView<&Mesh<G>, G>> {
        self.opposite
            .map(|opposite| EdgeView::new(self.mesh.as_ref(), opposite))
    }

    pub(in graph) fn into_raw_opposite_edge(self) -> Option<Self> {
        let opposite = self.opposite;
        let mesh = self.mesh;
        opposite.map(|opposite| EdgeView::new(mesh, opposite))
    }

    pub(in graph) fn raw_next_edge(&self) -> Option<EdgeView<&Mesh<G>, G>> {
        self.next
            .map(|next| EdgeView::new(self.mesh.as_ref(), next))
    }

    pub(in graph) fn into_raw_next_edge(self) -> Option<Self> {
        let next = self.next;
        let mesh = self.mesh;
        next.map(|next| EdgeView::new(mesh, next))
    }

    pub(in graph) fn raw_previous_edge(&self) -> Option<EdgeView<&Mesh<G>, G>> {
        self.previous
            .map(|previous| EdgeView::new(self.mesh.as_ref(), previous))
    }

    pub(in graph) fn into_raw_previous_edge(self) -> Option<Self> {
        let previous = self.previous;
        let mesh = self.mesh;
        previous.map(|previous| EdgeView::new(mesh, previous))
    }
}

impl<M, G> EdgeView<M, G>
where
    M: AsRef<Mesh<G>> + AsMut<Mesh<G>>,
    G: Geometry,
{
    pub fn opposite_edge_mut(&mut self) -> OrphanEdgeView<G> {
        self.raw_opposite_edge_mut().unwrap()
    }

    pub fn next_edge_mut(&mut self) -> OrphanEdgeView<G> {
        self.raw_next_edge_mut().unwrap()
    }

    pub fn previous_edge_mut(&mut self) -> OrphanEdgeView<G> {
        self.raw_previous_edge_mut().unwrap()
    }

    pub fn source_vertex_mut(&mut self) -> OrphanVertexView<G> {
        let (vertex, _) = self.key().to_vertex_keys();
        self.mesh.as_mut().orphan_vertex_mut(vertex).unwrap()
    }

    pub fn destination_vertex_mut(&mut self) -> OrphanVertexView<G> {
        let vertex = self.vertex;
        self.mesh.as_mut().orphan_vertex_mut(vertex).unwrap()
    }

    pub fn face_mut(&mut self) -> Option<OrphanFaceView<G>> {
        let face = self.face;
        face.map(move |face| self.mesh.as_mut().orphan_face_mut(face).unwrap())
    }

    pub fn boundary_edge_mut(&mut self) -> Option<OrphanEdgeView<G>> {
        use BoolExt;

        if self.is_boundary_edge() {
            Some(self.mesh.as_mut().orphan_edge_mut(self.key).unwrap())
        }
        else {
            self.opposite_edge()
                .is_boundary_edge()
                .into_some(self.opposite_edge_mut())
        }
    }

    // TODO: Rename this to something like "extend". It is very similar to
    //       `extrude`. Terms like "join" or "merge" are better suited for
    //       directly joining two adjacent faces over a shared edge.
    pub fn join(mut self, edge: EdgeKey) -> Result<Self, Error> {
        if self.mesh.as_ref().edge(edge).is_none() {
            return Err(GraphError::TopologyNotFound.into());
        }
        let (a, b) = self.key().to_vertex_keys();
        let (c, d) = edge.to_vertex_keys();
        // At this point, we can assume the points a, b, c, and d exist in the
        // mesh. Before mutating the mesh, ensure that existing interior edges
        // are boundaries.
        for edge in [a, b, c, d]
            .perimeter()
            .flat_map(|ab| self.mesh.as_ref().edge(ab.into()))
        {
            if !edge.is_boundary_edge() {
                return Err(GraphError::TopologyConflict.into());
            }
        }
        // Insert a quad joining the edges. These operations should not fail;
        // unwrap their results.
        let edge = self.geometry.clone();
        let face = self.opposite_edge()
            .face()
            .map(|face| face.geometry.clone())
            .unwrap_or_else(Default::default);
        // TODO: Split the face to form triangles.
        Mutation::immediate(self.mesh.as_mut())
            .insert_face(&[a, b, c, d], (edge, face))
            .unwrap();
        Ok(EdgeView::new(self.mesh, (c, d).into()))
    }

    #[allow(dead_code)]
    fn remove(self) -> Result<M, Error> {
        let EdgeView { mut mesh, key, .. } = self;
        Mutation::immediate(mesh.as_mut()).remove_edge(key)?;
        Ok(mesh)
    }

    // Resolve the `M` parameter to a concrete reference.
    #[allow(dead_code)]
    fn with_mesh_mut(&mut self) -> EdgeView<&mut Mesh<G>, G> {
        EdgeView::new(self.mesh.as_mut(), self.key)
    }
}

/// Raw API.
impl<M, G> EdgeView<M, G>
where
    M: AsRef<Mesh<G>> + AsMut<Mesh<G>>,
    G: Geometry,
{
    pub(in graph) fn raw_opposite_edge_mut(&mut self) -> Option<OrphanEdgeView<G>> {
        let opposite = self.opposite;
        opposite.map(move |opposite| self.mesh.as_mut().orphan_edge_mut(opposite).unwrap())
    }

    pub(in graph) fn raw_next_edge_mut(&mut self) -> Option<OrphanEdgeView<G>> {
        let next = self.next;
        next.map(move |next| self.mesh.as_mut().orphan_edge_mut(next).unwrap())
    }

    pub(in graph) fn raw_previous_edge_mut(&mut self) -> Option<OrphanEdgeView<G>> {
        let previous = self.previous;
        previous.map(move |previous| self.mesh.as_mut().orphan_edge_mut(previous).unwrap())
    }
}

impl<M, G> EdgeView<M, G>
where
    M: AsRef<Mesh<G>>,
    G: EdgeMidpoint + Geometry,
{
    pub fn midpoint(&self) -> Result<G::Midpoint, Error> {
        G::midpoint(self.with_mesh_ref())
    }
}

impl<M, G> EdgeView<M, G>
where
    M: AsRef<Mesh<G>> + AsMut<Mesh<G>>,
    G: EdgeMidpoint + Geometry,
    G::Vertex: AsPosition,
{
    pub fn split(mut self) -> Result<VertexView<M, G>, Error>
    where
        G: EdgeMidpoint<Midpoint = VertexPosition<G>>,
    {
        // Insert a new vertex at the midpoint.
        let m = {
            let mut m = self.source_vertex().geometry.clone();
            *m.as_position_mut() = self.midpoint()?;
            // This is the point of no return; the mesh has been mutated.
            Mutation::immediate(self.mesh.as_mut()).insert_vertex(m)
        };
        // TODO: This will not attempt to split the opposite edge if it does
        //       not exist. How should this tend to work? Should mutations like
        //       this panic if the mesh is inconsistent, or should mutations
        //       avoid panics? Will this kind of mutation be used internally
        //       when a mesh is in an intermediate state?
        // Get both half-edges to be split.
        let edge = self.key();
        let opposite = self.raw_opposite_edge().map(|opposite| opposite.key());
        let mut mesh = self.mesh;
        // Split the half-edges. This should not fail; unwrap the results.
        Self::split_half_at(&mut mesh, edge, m).unwrap();
        if let Some(opposite) = opposite {
            Self::split_half_at(&mut mesh, opposite, m).unwrap();
        }
        Ok(VertexView::new(mesh, m))
    }

    fn split_half_at(
        mesh: &mut M,
        edge: EdgeKey,
        m: VertexKey,
    ) -> Result<(EdgeKey, EdgeKey), Error> {
        // Remove the edge and insert two truncated edges in its place.
        let (a, b) = edge.to_vertex_keys();
        let (source, am, mb) = {
            let mut mutation = Mutation::immediate(mesh.as_mut());
            let source = mutation.remove_edge(edge).unwrap();
            let am = mutation.insert_edge((a, m), source.geometry.clone())?;
            let mb = mutation.insert_edge((m, b), source.geometry.clone())?;
            (source, am, mb)
        };
        // Connect the new edges to each other and their leading edges.
        {
            let mut edge = mesh.as_mut().edge_mut(am).unwrap();
            edge.next = Some(mb);
            edge.previous = source.previous;
            edge.face = source.face
        }
        {
            let mut edge = mesh.as_mut().edge_mut(mb).unwrap();
            edge.next = source.next;
            edge.previous = Some(am);
            edge.face = source.face;
        }
        if let Some(pa) = source.previous {
            mesh.as_mut().edge_mut(pa).unwrap().next = Some(am);
        }
        if let Some(bn) = source.next {
            mesh.as_mut().edge_mut(bn).unwrap().previous = Some(mb);
        }
        // Update the associated face, if any, because it may refer to the
        // removed edge.
        if let Some(face) = source.face {
            mesh.as_mut().face_mut(face).unwrap().edge = am;
        }
        Ok((am, mb))
    }
}

impl<M, G> EdgeView<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry + EdgeLateral,
{
    pub fn lateral(&self) -> Result<G::Lateral, Error> {
        G::lateral(self.with_mesh_ref())
    }
}

impl<M, G> EdgeView<M, G>
where
    M: AsRef<Mesh<G>> + AsMut<Mesh<G>>,
    G: Geometry + EdgeLateral,
    G::Vertex: AsPosition,
{
    pub fn extrude<T>(mut self, distance: T) -> Result<Self, Error>
    where
        G::Lateral: Mul<T>,
        ScaledEdgeLateral<G, T>: Clone,
        VertexPosition<G>: Add<ScaledEdgeLateral<G, T>, Output = VertexPosition<G>> + Clone,
    {
        if !self.is_boundary_edge() {
            return Err(GraphError::TopologyConflict.into());
        }
        // Insert new vertices with the specified translation.
        let (c, d) = {
            let mut c = self.destination_vertex().geometry.clone();
            let mut d = self.source_vertex().geometry.clone();
            // Clone the geometry and translate it using the lateral normal,
            // then insert the new vertex geometry and yield the vertex keys.
            let translation = self.lateral()? * distance;
            *c.as_position_mut() = c.as_position().clone() + translation.clone();
            *d.as_position_mut() = d.as_position().clone() + translation;
            let mut mutation = Mutation::immediate(self.mesh.as_mut());
            (
                // This is the point of no return; the mesh has been mutated.
                // Unwrap results.
                mutation.insert_vertex(c),
                mutation.insert_vertex(d),
            )
        };
        let edge = self.geometry.clone();
        let cd = Mutation::immediate(self.mesh.as_mut())
            .insert_edge((c, d), edge)
            .unwrap();
        Ok(self.join(cd).unwrap())
    }
}

impl<M, G> AsRef<EdgeView<M, G>> for EdgeView<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    fn as_ref(&self) -> &EdgeView<M, G> {
        self
    }
}

impl<M, G> AsMut<EdgeView<M, G>> for EdgeView<M, G>
where
    M: AsRef<Mesh<G>> + AsMut<Mesh<G>>,
    G: Geometry,
{
    fn as_mut(&mut self) -> &mut EdgeView<M, G> {
        self
    }
}

impl<M, G> Deref for EdgeView<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    type Target = Edge<G>;

    fn deref(&self) -> &Self::Target {
        self.mesh.as_ref().edges.get(&self.key).unwrap()
    }
}

impl<M, G> DerefMut for EdgeView<M, G>
where
    M: AsRef<Mesh<G>> + AsMut<Mesh<G>>,
    G: Geometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.mesh.as_mut().edges.get_mut(&self.key).unwrap()
    }
}

impl<M, G> Clone for EdgeView<M, G>
where
    M: AsRef<Mesh<G>> + Clone,
    G: Geometry,
{
    fn clone(&self) -> Self {
        EdgeView {
            mesh: self.mesh.clone(),
            key: self.key,
            phantom: PhantomData,
        }
    }
}

impl<M, G> Copy for EdgeView<M, G>
where
    M: AsRef<Mesh<G>> + Copy,
    G: Geometry,
{
}

impl<M, G> View<M, G> for EdgeView<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    type Topology = Edge<G>;

    fn from_mesh(mesh: M, key: <Self::Topology as Topological>::Key) -> Self {
        EdgeView::new(mesh, key)
    }
}

/// Do **not** use this type directly. Use `OrphanEdgeMut` instead.
///
/// This type is only re-exported so that its members are shown in
/// documentation. See this issue:
/// <https://github.com/rust-lang/rust/issues/39437>
pub struct OrphanEdgeView<'a, G>
where
    G: 'a + Geometry,
{
    key: EdgeKey,
    edge: &'a mut Edge<G>,
}

impl<'a, G> OrphanEdgeView<'a, G>
where
    G: 'a + Geometry,
{
    pub(in graph) fn new(edge: &'a mut Edge<G>, key: EdgeKey) -> Self {
        OrphanEdgeView {
            key: key,
            edge: edge,
        }
    }

    pub fn key(&self) -> EdgeKey {
        self.key
    }
}

impl<'a, G> Deref for OrphanEdgeView<'a, G>
where
    G: 'a + Geometry,
{
    type Target = <Self as OrphanView<'a, G>>::Topology;

    fn deref(&self) -> &Self::Target {
        &*self.edge
    }
}

impl<'a, G> DerefMut for OrphanEdgeView<'a, G>
where
    G: 'a + Geometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.edge
    }
}

impl<'a, G> OrphanView<'a, G> for OrphanEdgeView<'a, G>
where
    G: 'a + Geometry,
{
    type Topology = Edge<G>;

    fn from_topology(
        topology: &'a mut Self::Topology,
        key: <Self::Topology as Topological>::Key,
    ) -> Self {
        OrphanEdgeView::new(topology, key)
    }
}

pub struct EdgeKeyTopology {
    key: EdgeKey,
    vertices: (VertexKey, VertexKey),
}

impl EdgeKeyTopology {
    fn new(edge: EdgeKey, vertices: (VertexKey, VertexKey)) -> Self {
        EdgeKeyTopology {
            key: edge,
            vertices: vertices,
        }
    }

    pub fn key(&self) -> EdgeKey {
        self.key
    }

    pub fn vertices(&self) -> (VertexKey, VertexKey) {
        self.vertices
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Point3;

    use generate::*;
    use graph::*;
    use graph::storage::Key;

    #[test]
    fn extrude_edge() {
        let mut mesh = Mesh::<Point3<f32>>::from_raw_buffers(
            vec![0, 1, 2, 3],
            vec![
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ],
            4,
        ).unwrap();
        let key = mesh.edges()
            .flat_map(|edge| edge.into_boundary_edge())
            .nth(0)
            .unwrap()
            .key();
        mesh.edge_mut(key).unwrap().extrude(1.0).unwrap();

        assert_eq!(14, mesh.edge_count());
        assert_eq!(2, mesh.face_count());
    }

    #[test]
    fn join_edges() {
        // Construct a mesh with two independent quads.
        let mut mesh = Mesh::<Point3<f32>>::from_raw_buffers(
            vec![0, 1, 2, 3, 4, 5, 6, 7],
            vec![
                (-2.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0), // 1
                (-1.0, 1.0, 0.0), // 2
                (-2.0, 1.0, 0.0),
                (1.0, 0.0, 0.0), // 4
                (2.0, 0.0, 0.0),
                (2.0, 1.0, 0.0),
                (1.0, 1.0, 0.0), // 7
            ],
            4,
        ).unwrap();
        // TODO: This is fragile. It would probably be best for `Mesh` to
        //       provide a more convenient way to search for topology.
        // Construct the keys for the nearby edges.
        let source = (VertexKey::from(Key::new(2)), VertexKey::from(Key::new(1))).into();
        let destination = (VertexKey::from(Key::new(4)), VertexKey::from(Key::new(7))).into();
        mesh.edge_mut(source).unwrap().join(destination).unwrap();

        assert_eq!(20, mesh.edge_count());
        assert_eq!(3, mesh.face_count());
    }

    #[test]
    fn split_composite_edge() {
        let (indeces, vertices) = cube::Cube::new()
            .polygons_with_position() // 6 quads, 24 vertices.
            .flat_index_vertices(HashIndexer::default());
        let mut mesh = Mesh::<Point3<f32>>::from_raw_buffers(indeces, vertices, 4).unwrap();
        let key = mesh.edges().nth(0).unwrap().key();
        let vertex = mesh.edge_mut(key).unwrap().split().unwrap();

        assert_eq!(5, vertex.outgoing_edge().face().unwrap().edges().count());
        assert_eq!(
            5,
            vertex
                .outgoing_edge()
                .opposite_edge()
                .face()
                .unwrap()
                .edges()
                .count()
        );
    }
}
