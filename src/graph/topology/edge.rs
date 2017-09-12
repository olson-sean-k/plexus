use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use graph::geometry::Geometry;
use graph::mesh::{Edge, Mesh};
use graph::storage::EdgeKey;

#[derive(Clone, Copy)]
pub struct EdgeView<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    mesh: M,
    pub key: EdgeKey,
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

    pub fn as_opposite(&self) -> Option<EdgeView<&Mesh<G>, G>> {
        self.opposite
            .map(|opposite| EdgeView::new(self.mesh.as_ref(), opposite))
    }

    pub fn into_opposite(self) -> Option<Self> {
        let opposite = self.opposite;
        let mesh = self.mesh;
        opposite.map(|opposite| EdgeView::new(mesh, opposite))
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
    pub fn as_opposite_mut(&mut self) -> Option<EdgeView<&mut Mesh<G>, G>> {
        let opposite = self.opposite;
        let mesh = self.mesh.as_mut();
        opposite.map(|opposite| EdgeView::new(mesh, opposite))
    }

    // Resolve the `M` parameter to a concrete reference.
    #[allow(dead_code)]
    fn with_mesh_mut(&mut self) -> EdgeView<&mut Mesh<G>, G> {
        EdgeView::new(self.mesh.as_mut(), self.key)
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

// There's no need to abstract over mutability for this type. For immutable
// refs, there is no need for an orphan type. Moreover, it is not possible to
// implement `AsRef` and `AsMut` for all types that implement `Geometry`.
pub struct OrphanEdgeView<'a, G>
where
    G: 'a + Geometry,
{
    pub key: EdgeKey,
    // The name `geometry` mirrors the `geometry` field of `Edge`, to which
    // `EdgeView` derefs.
    pub geometry: &'a mut G::Edge,
}

impl<'a, G> OrphanEdgeView<'a, G>
where
    G: 'a + Geometry,
{
    pub(crate) fn new(geometry: &'a mut G::Edge, edge: EdgeKey) -> Self {
        OrphanEdgeView {
            key: edge,
            geometry: geometry,
        }
    }
}
