use num::Float;
use std::marker::PhantomData;
use std::ops::{Add, Deref, DerefMut, Mul, Sub};

use graph::geometry::{AsPosition, Cross, Geometry, Normalize};
use graph::mesh::{Face, Mesh};
use graph::storage::{EdgeKey, FaceKey, VertexKey};
use graph::topology::EdgeView;

pub type FaceRef<'a, G> = FaceView<&'a Mesh<G>, G>;
pub type FaceMut<'a, G> = FaceView<&'a mut Mesh<G>, G>;

#[derive(Clone, Copy)]
pub struct FaceView<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    mesh: M,
    pub key: FaceKey,
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

    pub fn edges(&self) -> EdgeCirculator<&Mesh<G>, G> {
        EdgeCirculator::new(self.with_mesh_ref())
    }

    pub fn faces(&self) -> FaceCirculator<&Mesh<G>, G> {
        FaceCirculator::new(self.with_mesh_ref())
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
    pub fn faces_mut(&mut self) -> FaceCirculator<&mut Mesh<G>, G> {
        FaceCirculator::new(self.with_mesh_mut())
    }

    // Resolve the `M` parameter to a concrete reference.
    fn with_mesh_mut(&mut self) -> FaceView<&mut Mesh<G>, G> {
        FaceView::new(self.mesh.as_mut(), self.key)
    }
}

// The elaborate type constraints state that positions stored in the vertex
// geometry can be used to compute a normal of a face using vector subtraction,
// cross product, normalization, vector addition, and vector scaling.
#[cfg_attr(rustfmt, rustfmt_skip)]
impl<M, G> FaceView<M, G>
where
    M: AsRef<Mesh<G>> + AsMut<Mesh<G>>,
    G: Geometry,
    G::Vertex: AsPosition + Clone,
    G::Edge: Clone,
    G::Face: Clone,
    // Constraints on the vertex geometry when converted to a position:
    //
    // 1. Supports cloning.
    // 2. Supports subtraction with itself.
    <G::Vertex as AsPosition>::Target: Clone + Sub,
    // Constraints on the output of subtraction of vertex geometry when
    // converted to a position:
    //
    // 1. Supports the cross product.
    <<G::Vertex as AsPosition>::Target as Sub>::Output: Cross,
    // Constraints on the output of the cross product of subtraction of vertex
    // geometry when converted to a position:
    //
    // 1. Supports normalization.
    <<<G::Vertex as AsPosition>::Target as Sub>::Output as Cross>::Output: Normalize,
{
    pub fn extrude<F>(self, distance: F) -> Result<Self, ()>
    where
        // See `extrude_vertex_geometry`.
        F: Float,
        <G::Vertex as AsPosition>::Target: Add<<<<<G::Vertex as AsPosition>::Target as Sub>::Output as Cross>::Output as Mul<F>>::Output, Output = <G::Vertex as AsPosition>::Target>,
        <<<G::Vertex as AsPosition>::Target as Sub>::Output as Cross>::Output: Mul<F>,
        <<<<G::Vertex as AsPosition>::Target as Sub>::Output as Cross>::Output as Mul<F>>::Output: Clone,
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
        // See `extrude_vertex_geometry`.
        F: Float,
        <G::Vertex as AsPosition>::Target: Add<<<<<G::Vertex as AsPosition>::Target as Sub>::Output as Cross>::Output as Mul<F>>::Output, Output = <G::Vertex as AsPosition>::Target>,
        <<<G::Vertex as AsPosition>::Target as Sub>::Output as Cross>::Output: Mul<F>,
        <<<<G::Vertex as AsPosition>::Target as Sub>::Output as Cross>::Output as Mul<F>>::Output: Clone,
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
            let vertices = geometry.into_iter()
                .map(|vertex| (vertex.0, mesh.insert_vertex(vertex.1))).collect::<Vec<_>>();
            let edges = vertices.iter().enumerate().map(|(index, &(_, a))| {
                let b = vertices[(index + 1) % vertices.len()].1;
                mesh.insert_edge((a, b), edge.clone()).unwrap()
            }).collect::<Vec<_>>();
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
        F: Float,
        // Constraints on the vertex geometry when converted to a position:
        //
        // 1. Supports addition with the output of multiplication with `F` of
        //    the cross product of subtraction with itself.
        // 2. The output of the above addition is itself.
        <G::Vertex as AsPosition>::Target: Add<<<<<G::Vertex as AsPosition>::Target as Sub>::Output as Cross>::Output as Mul<F>>::Output, Output = <G::Vertex as AsPosition>::Target>,
        // Constraints on the output of the cross product of subtraction of
        // vertex geometry when converted to a position:
        //
        // 1. Supports multiplication with `F`.
        <<<G::Vertex as AsPosition>::Target as Sub>::Output as Cross>::Output: Mul<F>,
        // Constraints on the output of multiplication with `F` of the cross
        // product of subtraction of vertex geometry when converted to a
        // position:
        //
        // 1. Supports cloning.
        <<<<G::Vertex as AsPosition>::Target as Sub>::Output as Cross>::Output as Mul<F>>::Output: Clone,
    {
        let mesh = self.mesh.as_ref();
        let edges = self.edges().collect::<Vec<_>>();
        if edges.len() < 3 {
            return Err(());
        }
        let (a, b, c) = (
            mesh.vertices.get(&edges[0].vertex).unwrap(),
            mesh.vertices.get(&edges[1].vertex).unwrap(),
            mesh.vertices.get(&edges[2].vertex).unwrap(),
        );
        let ab = a.geometry.as_position().clone() - b.geometry.as_position().clone();
        let bc = b.geometry.as_position().clone() - c.geometry.as_position().clone();
        let translation = ab.cross(bc).normalize() * distance;
        Ok(edges.into_iter().map(|edge| {
            let mut geometry = mesh.vertices.get(&edge.vertex).unwrap().geometry.clone();
            *geometry.as_position_mut() = geometry.as_position().clone() + translation.clone();
            (edge.vertex, geometry)
        }).collect())
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
    type Target = Face<G::Face>;

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

pub type OrphanFaceMut<'a, G> = OrphanFaceView<'a, G>;

// There's no need to abstract over mutability for this type. For immutable
// refs, there is no need for an orphan type. Moreover, it is not possible to
// implement `AsRef` and `AsMut` for all types that implement `Geometry`.
pub struct OrphanFaceView<'a, G>
where
    G: 'a + Geometry,
{
    pub key: FaceKey,
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

pub struct FaceCirculator<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    face: FaceView<M, G>,
    edge: Option<EdgeKey>,
    breadcrumb: Option<EdgeKey>,
}

impl<M, G> FaceCirculator<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    fn new(face: FaceView<M, G>) -> Self {
        let edge = face.edge;
        FaceCirculator {
            face: face,
            edge: Some(edge),
            breadcrumb: Some(edge),
        }
    }

    fn next(&mut self) -> Option<FaceKey> {
        let mesh = self.face.mesh.as_ref();
        while let Some(edge) = self.edge.map(|edge| mesh.edges.get(&edge).unwrap()) {
            self.edge = edge.next;
            if let Some(face) = edge.opposite
                .map(|opposite| mesh.edges.get(&opposite).unwrap())
                .and_then(|opposite| opposite.face)
            {
                return if let Some(_) = self.breadcrumb {
                    if self.breadcrumb == edge.next {
                        self.breadcrumb = None;
                    }
                    Some(face)
                }
                else {
                    None
                };
            }
            else {
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
        <FaceCirculator<_, _>>::next(self).map(|face| FaceView::new(self.face.mesh, face))
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
                    let face = mem::transmute::<_, &'a mut Face<G::Face>>(
                        self.face.mesh.faces.get_mut(&face).unwrap(),
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
    use r32;
    use generate::*;
    use graph::*;

    #[test]
    fn circulate_over_edges() {
        let mesh = sphere::UVSphere::<f32>::with_unit_radius(3, 2)
            .spatial_polygons() // 6 triangles, 18 vertices.
            .map_vertices(|vertex| vertex.into_hash())
            .triangulate()
            .collect::<Mesh<(r32, r32, r32)>>();
        // TODO: Provide a way to get a key for the faces in the mesh. Using
        //       `default` only works if the initial face has not been removed.
        let face = mesh.face(FaceKey::default()).unwrap();

        // All faces should be triangles and should have three edges.
        assert_eq!(3, face.edges().count());
    }

    #[test]
    fn circulate_over_faces() {
        let mesh = sphere::UVSphere::<f32>::with_unit_radius(3, 2)
            .spatial_polygons() // 6 triangles, 18 vertices.
            .map_vertices(|vertex| vertex.into_hash())
            .triangulate()
            .collect::<Mesh<(r32, r32, r32)>>();
        // TODO: Provide a way to get a key for the faces in the mesh. Using
        //       `default` only works if the initial face has not been removed.
        let face = mesh.face(FaceKey::default()).unwrap();

        // No matter which face is selected, it should have three neighbors.
        assert_eq!(3, face.faces().count());
    }

    #[test]
    fn extrude_face() {
        use nalgebra::Point3;

        let mut mesh: Mesh<Point3<f32>> = sphere::UVSphere::<f32>::with_unit_radius(3, 2)
            .spatial_polygons() // 6 triangles, 18 vertices.
            .map_vertices(|vertex| vertex.into_hash())
            .triangulate()
            .collect::<Mesh<(r32, r32, r32)>>()
            .into_geometry();
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
