//! Topological views into a `Mesh`.
//!
//! This module provides topological traits and views into a [`Mesh`]. Views
//! allow the topology of a graph to be traversed and mutated in a consistent
//! way, and also expose the geometry of the graph.
//!
//! Views behave like references, and are exposed as `...Ref`, `...Mut`, and
//! `Orphan...Mut` types (immutable, mutable, and orphan views, respectively)
//! summarized below:
//!
//! | Type      | Name           | Traversal | Arity | Geometry  | Topology  |
//! |-----------|----------------|-----------|-------|-----------|-----------|
//! | Immutable | `...Ref`       | Yes       | Many  | Immutable | Immutable |
//! | Mutable   | `...Mut`       | Neighbors | One   | Mutable   | Mutable   |
//! | Orphan    | `Orphan...Mut` | No        | Many  | Mutable   | N/A       |
//!
//! Note that it is not possible to get mutable views from another mutable view
//! via a traversal, because a mutation may alter the topology and invalidate
//! the originating view. This also means that mutable operations will always
//! consume `self`. In general, an immutable traversal of topology can be used
//! to collect keys that are later used to query and mutate the target
//! topology.

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

use graph::Mesh;
use graph::geometry::{Attribute, Geometry};
use graph::storage::OpaqueKey;

mod edge;
mod face;
mod vertex;

pub use self::edge::{EdgeKeyTopology, EdgeView, OrphanEdgeView};
pub use self::face::{FaceKeyTopology, FaceView, OrphanFaceView};
pub use self::vertex::{OrphanVertexView, VertexView};

pub type EdgeRef<'a, G> = EdgeView<&'a Mesh<G>, G>;
pub type EdgeMut<'a, G> = EdgeView<&'a mut Mesh<G>, G>;
pub type OrphanEdgeMut<'a, G> = OrphanEdgeView<'a, G>;

pub type FaceRef<'a, G> = FaceView<&'a Mesh<G>, G>;
pub type FaceMut<'a, G> = FaceView<&'a mut Mesh<G>, G>;
pub type OrphanFaceMut<'a, G> = OrphanFaceView<'a, G>;

pub type VertexRef<'a, G> = VertexView<&'a Mesh<G>, G>;
pub type VertexMut<'a, G> = VertexView<&'a mut Mesh<G>, G>;
pub type OrphanVertexMut<'a, G> = OrphanVertexView<'a, G>;

pub trait Topological {
    type Key: OpaqueKey;
    type Attribute: Attribute;
}

pub trait View<M, G>
where
    M: AsRef<Mesh<G>>,
    G: Geometry,
{
    type Topology: Topological;

    fn of(mesh: M, key: <Self::Topology as Topological>::Key) -> Self;
}

pub trait OrphanView<'a, G>
where
    G: Geometry,
{
    type Topology: Topological;

    // TODO: It would probably be better for this to accept a reference to the
    //       attribute instead of the topology, but it is difficult to isolate
    //       this type with the given traits and would require more machinery.
    //       For now, it is expected that types implementing this function will
    //       take a reference to the geometry within the given topology.
    fn of(topology: &'a mut Self::Topology, key: <Self::Topology as Topological>::Key) -> Self;
}
