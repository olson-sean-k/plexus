use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use graph::geometry::Geometry;
use graph::mesh::{Edge, Mesh, Vertex};
use graph::storage::{EdgeKey, VertexKey};
use graph::topology::{EdgeView, OrphanEdgeView};

#[derive(Clone, Copy)]
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
    pub(crate) fn new(mesh: M, vertex: VertexKey) -> Self {
        VertexView {
            mesh: mesh,
            key: vertex,
            phantom: PhantomData,
        }
    }

    pub fn key(&self) -> VertexKey {
        self.key
    }

    pub fn as_outgoing_edge(&self) -> Option<EdgeView<&Mesh<G>, G>> {
        self.edge
            .map(|edge| EdgeView::new(self.mesh.as_ref(), edge))
    }

    pub fn into_outgoing_edge(self) -> Option<EdgeView<M, G>> {
        let edge = self.edge;
        let mesh = self.mesh;
        edge.map(|edge| EdgeView::new(mesh, edge))
    }

    pub fn edges(&self) -> EdgeCirculator<&Mesh<G>, G> {
        EdgeCirculator::new(self.with_mesh_ref())
    }

    // Resolve the `M` parameter to a concrete reference.
    fn with_mesh_ref(&self) -> VertexView<&Mesh<G>, G> {
        VertexView::new(self.mesh.as_ref(), self.key)
    }
}

impl<M, G> VertexView<M, G>
where
    M: AsRef<Mesh<G>> + AsMut<Mesh<G>>,
    G: Geometry,
{
    pub fn as_outgoing_edge_mut(&mut self) -> Option<OrphanEdgeView<G>> {
        let edge = self.edge;
        edge.map(move |edge| {
            OrphanEdgeView::new(
                &mut self.mesh.as_mut().edges.get_mut(&edge).unwrap().geometry,
                edge,
            )
        })
    }

    pub fn edges_mut(&mut self) -> EdgeCirculator<&mut Mesh<G>, G> {
        EdgeCirculator::new(self.with_mesh_mut())
    }

    // Resolve the `M` parameter to a concrete reference.
    fn with_mesh_mut(&mut self) -> VertexView<&mut Mesh<G>, G> {
        VertexView::new(self.mesh.as_mut(), self.key)
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
            .map(|outgoing| {
                self.vertex.mesh.as_ref().edges.get(&outgoing).unwrap()
            })
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
                        self.vertex.mesh.edges.get_mut(&edge).unwrap(),
                    );
                    &mut edge.geometry
                }
            };
            OrphanEdgeView::new(geometry, edge)
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
        // TODO: Provide a way to get a key for the vertices in the mesh. Using
        //       `default` only works if the initial vertex has not been
        //       removed.
        let vertex = mesh.vertex(VertexKey::default()).unwrap();

        // All faces should be triangles and all vertices should have three
        // incoming edges.
        assert_eq!(3, vertex.edges().count());
    }
}
