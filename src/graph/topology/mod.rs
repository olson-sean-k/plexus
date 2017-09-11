mod edge;
mod face;

// TODO: Generalize the pairing of a ref to a mesh and a key for topology
//       within the mesh.

// This code assumes that any keys for topological structures in the mesh are
// valid (hence the `unwrap` calls), which is very important for `Deref`.
// Topological mutations using views like `FaceView` are dangerous if they do
// not consume `self`. If these views can be used to mutate that data, then
// they can also invalidate these constraints and cause panics. Any mutating
// functions should consume the view.

pub use self::edge::{EdgeMut, EdgeRef, EdgeView};
pub use self::face::{FaceMut, FaceRef, FaceView, OrphanFaceMut};
