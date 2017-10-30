use std::marker::PhantomData;
use std::ops::{Add, Deref, DerefMut, Mul};

use geometry::Geometry;
use geometry::convert::AsPosition;
use graph::geometry::LateralNormal;
use graph::geometry::alias::{ScaledLateralNormal, VertexPosition};
use graph::mesh::{Edge, Mesh};
use graph::storage::{EdgeKey, VertexKey};
use graph::topology::{FaceView, OrphanFaceView, OrphanVertexView, OrphanView, Topological,
                      VertexView, View};

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
    pub(crate) fn new(mesh: M, edge: EdgeKey) -> Self {
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
        EdgeKeyTopology::new(self.key, (self.vertex, self.next().map(|edge| edge.vertex)))
    }

    pub fn opposite(&self) -> Option<EdgeView<&Mesh<G>, G>> {
        self.opposite
            .map(|opposite| EdgeView::new(self.mesh.as_ref(), opposite))
    }

    pub fn into_opposite(self) -> Option<Self> {
        let opposite = self.opposite;
        let mesh = self.mesh;
        opposite.map(|opposite| EdgeView::new(mesh, opposite))
    }

    pub fn next(&self) -> Option<EdgeView<&Mesh<G>, G>> {
        self.next
            .map(|next| EdgeView::new(self.mesh.as_ref(), next))
    }

    pub fn into_next(self) -> Option<Self> {
        let next = self.next;
        let mesh = self.mesh;
        next.map(|next| EdgeView::new(mesh, next))
    }

    pub fn vertex(&self) -> VertexView<&Mesh<G>, G> {
        VertexView::new(self.mesh.as_ref(), self.vertex)
    }

    pub fn into_vertex(self) -> VertexView<M, G> {
        let vertex = self.vertex;
        let mesh = self.mesh;
        VertexView::new(mesh, vertex)
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

    // Resolve the `M` parameter to a concrete reference.
    #[allow(dead_code)]
    fn with_mesh_ref(&self) -> EdgeView<&Mesh<G>, G> {
        EdgeView::new(self.mesh.as_ref(), self.key)
    }
}

impl<M, G> EdgeView<M, G>
where
    M: AsRef<Mesh<G>> + AsMut<Mesh<G>>,
    G: Geometry,
{
    pub fn opposite_mut(&mut self) -> Option<OrphanEdgeView<G>> {
        let opposite = self.opposite;
        opposite.map(move |opposite| {
            OrphanEdgeView::new(
                self.mesh.as_mut().edges.get_mut(&opposite).unwrap(),
                opposite,
            )
        })
    }

    pub fn next_mut(&mut self) -> Option<OrphanEdgeView<G>> {
        let next = self.next;
        next.map(move |next| {
            OrphanEdgeView::new(self.mesh.as_mut().edges.get_mut(&next).unwrap(), next)
        })
    }

    pub fn vertex_mut(&mut self) -> OrphanVertexView<G> {
        let vertex = self.vertex;
        OrphanVertexView::new(
            self.mesh.as_mut().vertices.get_mut(&vertex).unwrap(),
            vertex,
        )
    }

    pub fn face_mut(&mut self) -> Option<OrphanFaceView<G>> {
        let face = self.face;
        face.map(move |face| {
            OrphanFaceView::new(self.mesh.as_mut().faces.get_mut(&face).unwrap(), face)
        })
    }

    // Resolve the `M` parameter to a concrete reference.
    #[allow(dead_code)]
    fn with_mesh_mut(&mut self) -> EdgeView<&mut Mesh<G>, G> {
        EdgeView::new(self.mesh.as_mut(), self.key)
    }
}

impl<M, G> EdgeView<M, G>
where
    M: AsRef<Mesh<G>> + AsMut<Mesh<G>>,
    G: Geometry + LateralNormal,
    G::Vertex: AsPosition,
{
    pub fn extrude<T>(mut self, distance: T) -> Result<Self, ()>
    where
        G::Normal: Mul<T>,
        ScaledLateralNormal<G, T>: Clone,
        VertexPosition<G>: Add<ScaledLateralNormal<G, T>, Output = VertexPosition<G>> + Clone,
    {
        if self.opposite.is_some() {
            return Err(());
        }
        // Insert new vertices with the specified translation and get all
        // vertex keys.
        let (a, b, c, d) = {
            // Get the originating vertices and their geometry.
            let (a, ag, b, bg) = {
                let next = self.next().ok_or(())?;
                let a = self.vertex();
                let b = next.vertex();
                (a.key(), a.geometry.clone(), b.key(), b.geometry.clone())
            };
            // Clone the geometry and translate it using the lateral normal,
            // then insert the new vertex geometry and yield the vertex keys.
            let translation = G::normal(self.with_mesh_ref())? * distance;
            let mut cg = bg.clone();
            let mut dg = ag.clone();
            *cg.as_position_mut() = bg.as_position().clone() + translation.clone();
            *dg.as_position_mut() = ag.as_position().clone() + translation;
            (
                a,
                b,
                self.mesh.as_mut().insert_vertex(cg),
                self.mesh.as_mut().insert_vertex(dg),
            )
        };
        // Insert the edges and faces (two triangles forming a quad) and get
        // the extruded edge's key.
        let extrusion = {
            let face = self.face().ok_or(())?.geometry.clone();
            let mesh = self.mesh.as_mut();
            // Triangle of b-a-d.
            let ba = mesh.insert_edge((b, a), G::Edge::default())?;
            let ad = mesh.insert_edge((a, d), G::Edge::default())?;
            let db = mesh.insert_edge((d, b), G::Edge::default())?;
            // Triangle of b-d-c.
            let bd = mesh.insert_edge((b, d), G::Edge::default())?;
            let dc = mesh.insert_edge((d, c), G::Edge::default())?;
            let cb = mesh.insert_edge((c, b), G::Edge::default())?;
            mesh.insert_face(&[ba, ad, db], face.clone())?;
            mesh.insert_face(&[bd, dc, cb], face)?;
            dc
        };
        Ok(EdgeView::new(self.mesh, extrusion))
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
            key: self.key.clone(),
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
    pub(crate) fn new(edge: &'a mut Edge<G>, key: EdgeKey) -> Self {
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
    vertices: (VertexKey, Option<VertexKey>),
}

impl EdgeKeyTopology {
    fn new(edge: EdgeKey, vertices: (VertexKey, Option<VertexKey>)) -> Self {
        EdgeKeyTopology {
            key: edge,
            vertices: vertices,
        }
    }

    pub fn key(&self) -> EdgeKey {
        self.key
    }

    pub fn vertices(&self) -> (VertexKey, Option<VertexKey>) {
        self.vertices
    }
}
