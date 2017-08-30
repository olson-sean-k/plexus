use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use graph::geometry::Geometry;
use graph::mesh::{Face, Mesh};
use graph::storage::{EdgeKey, FaceKey};

// TODO: Generalize this pairing of a ref to a mesh and a key for topology
//       within the mesh.

// NOTE: Topological mutations using views like `FaceView` are dangerous as
//       this is written. This code assumes that any keys for topological
//       structure in the mesh are valid (hence the `unwrap` calls), which is
//       very important for `Deref`. If these views can be used to mutate that
//       data, then they can also invalidate this constraint and cause panics.

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
    pub(super) fn new(mesh: M, face: FaceKey) -> Self {
        FaceView {
            mesh: mesh,
            key: face,
            phantom: PhantomData,
        }
    }

    fn with_mesh_ref(&self) -> FaceView<&Mesh<G>, G> {
        FaceView::new(self.mesh.as_ref(), self.key)
    }
}

impl<'a, G> FaceView<&'a Mesh<G>, G>
where
    G: Geometry,
{
    pub fn faces(&self) -> FaceCirculator<&Mesh<G>, G> {
        FaceCirculator::new(self.with_mesh_ref())
    }
}

impl<M, G> FaceView<M, G>
where
    M: AsRef<Mesh<G>> + AsMut<Mesh<G>>,
    G: Geometry,
{
    fn with_mesh_mut(&mut self) -> FaceView<&mut Mesh<G>, G> {
        FaceView::new(self.mesh.as_mut(), self.key)
    }
}

impl<'a, G> FaceView<&'a mut Mesh<G>, G>
where
    G: Geometry,
{
    // TODO: Should this be named `faces_mut`? This is mutually exclusive with
    //       the non-mutable variant, but perhaps it would still be good to
    //       mark this as mutable.
    pub fn faces(&mut self) -> FaceCirculator<&mut Mesh<G>, G> {
        FaceCirculator::new(self.with_mesh_mut())
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
        self.mesh.as_ref().faces.get(self.key).unwrap()
    }
}

impl<M, G> DerefMut for FaceView<M, G>
where
    M: AsRef<Mesh<G>> + AsMut<Mesh<G>>,
    G: Geometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.mesh.as_mut().faces.get_mut(self.key).unwrap()
    }
}

pub struct FaceCirculator<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    face: FaceView<M, G>,
    edge: Option<EdgeKey>,
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
        }
    }

    fn next(&mut self) -> Option<FaceKey> {
        let mesh = self.face.mesh.as_ref();
        while let Some(edge) = self.edge.map(|edge| mesh.edges.get(edge).unwrap()) {
            self.edge = edge.next;
            if let Some(face) = edge.opposite
                .map(|opposite| mesh.edges.get(opposite).unwrap())
                .and_then(|opposite| opposite.face)
            {
                return Some(face);
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
    // TODO: One way to "unify" the output types of this iterator is to
    //       introduce an `OrphanFaceView` that holds a reference to the
    //       geometry but not the mesh. Code working with these structures
    //       would look nearly the same.
    //
    // This cannot be a `FaceView`, because that would alias the mutable
    // reference to the mesh. Instead, yield the key and a mutable reference to
    // the geometry data.
    type Item = (FaceKey, &'a mut G::Face);

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
                        self.face.mesh.faces.get_mut(face).unwrap(),
                    );
                    &mut face.geometry
                }
            };
            (face, geometry)
        })
    }
}
