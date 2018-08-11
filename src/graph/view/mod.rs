// This code assumes that any keys for topological structures in the mesh are
// valid (hence the `unwrap` calls), which is very important for `Deref`.
// Topological mutations using views are dangerous if they do not consume
// `self`. If these views can be used to mutate that data, then they can also
// invalidate these constraints and cause panics. Any mutating functions should
// consume the view.
//
// Similarly, toplogical mutations could invalidate views used to reach other
// views. This means that it is unsafe for a mutable view to yield another
// mutable view, because the second view may cause mutations that invalidate
// the first. Circulators effectively map from a mutable view to orphan views,
// for example. While `into` and immutable accessor functions are okay, mutable
// accessor functions MUST yield orphans (or not exist at all).

// TODO: Because circulators are not intended to be re-exported and using
//       dedicated types leads to more code and some duplication, it would be
//       preferable to use `impl Iterator<Item = ...>`.  However, due to an
//       unresolved issue, this severely limits the lifetime of iterators. This
//       has a drastic effect on the ergonomics of call chains that traverse a
//       graph, so dedicated types are used for now. See this issue:
//       https://github.com/rust-lang/rust/issues/50823

use graph::mesh::Mesh;

pub mod convert;
mod edge;
mod face;
mod vertex;

pub use self::edge::{EdgeKeyTopology, EdgeView, OrphanEdgeView};
pub use self::face::{FaceKeyTopology, FaceView, OrphanFaceView};
pub use self::vertex::{OrphanVertexView, VertexView};

pub type EdgeRef<'a, G> = EdgeView<&'a Mesh<G>, G>;
pub type EdgeMut<'a, G> = EdgeView<&'a mut Mesh<G>, G>;
pub type OrphanEdge<'a, G> = OrphanEdgeView<'a, G>;

pub type FaceRef<'a, G> = FaceView<&'a Mesh<G>, G>;
pub type FaceMut<'a, G> = FaceView<&'a mut Mesh<G>, G>;
pub type OrphanFace<'a, G> = OrphanFaceView<'a, G>;

pub type VertexRef<'a, G> = VertexView<&'a Mesh<G>, G>;
pub type VertexMut<'a, G> = VertexView<&'a mut Mesh<G>, G>;
pub type OrphanVertex<'a, G> = OrphanVertexView<'a, G>;

pub trait ConsistencyContract {}

pub struct Consistent;

impl ConsistencyContract for Consistent {}

pub struct Indeterminate;

impl ConsistencyContract for Indeterminate {}

pub trait Container {
    type Contract: ConsistencyContract;
}

impl<'a, T> Container for &'a T
where
    T: Container,
{
    type Contract = <T as Container>::Contract;
}

impl<'a, T> Container for &'a mut T
where
    T: Container,
{
    type Contract = <T as Container>::Contract;
}

pub trait Reborrow {
    type Target;

    fn reborrow(&self) -> &Self::Target;
}

pub trait ReborrowMut: Reborrow {
    fn reborrow_mut(&mut self) -> &mut Self::Target;
}

impl<'a, T> Reborrow for &'a T {
    type Target = T;

    fn reborrow(&self) -> &Self::Target {
        *self
    }
}

impl<'a, T> Reborrow for &'a mut T {
    type Target = T;

    fn reborrow(&self) -> &Self::Target {
        &**self
    }
}

impl<'a, T> ReborrowMut for &'a mut T {
    fn reborrow_mut(&mut self) -> &mut Self::Target {
        *self
    }
}
