use graph::Mesh;
use graph::geometry::Attribute;
use graph::storage::OpaqueKey;

mod edge;
mod face;
mod vertex;

// TODO: Generalize the pairing of a ref to a mesh and a key for topology
//       within the mesh.

// This code assumes that any keys for topological structures in the mesh are
// valid (hence the `unwrap` calls), which is very important for `Deref`.
// Topological mutations using views like `FaceView` are dangerous if they do
// not consume `self`. If these views can be used to mutate that data, then
// they can also invalidate these constraints and cause panics. Any mutating
// functions should consume the view.

pub use self::edge::{EdgeView, OrphanEdgeView};
pub use self::face::{FaceView, OrphanFaceView};
pub use self::vertex::VertexView;

pub type EdgeRef<'a, G> = EdgeView<&'a Mesh<G>, G>;
pub type EdgeMut<'a, G> = EdgeView<&'a mut Mesh<G>, G>;
pub type OrphanEdgeMut<'a, G> = OrphanEdgeView<'a, G>;

pub type FaceRef<'a, G> = FaceView<&'a Mesh<G>, G>;
pub type FaceMut<'a, G> = FaceView<&'a mut Mesh<G>, G>;
pub type OrphanFaceMut<'a, G> = OrphanFaceView<'a, G>;

pub type VertexRef<'a, G> = VertexView<&'a Mesh<G>, G>;
pub type VertexMut<'a, G> = VertexView<&'a mut Mesh<G>, G>;

pub trait Topological {
    type Key: OpaqueKey;
    type Attribute: Attribute;
}
