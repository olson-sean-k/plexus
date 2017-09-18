//! Topological views into a `Mesh`.
//!
//! This module provides topological traits and views into a `Mesh`. Views
//! allow the topology of a graph to be traversed and mutated in a consistent
//! way, and also expose the geometry of the graph.
//!
//! Views behave like references, and are exposed as `...Ref`, `...Mut`, and
//! `Orphan...Mut` types. Summarized below.
//!
//!   - `...Ref`
//!     Immutable reference to topology. Can be freely traversed, yielding
//!     arbitrary topology, but cannot mutate geometry or topology.
//!   - `...Mut`
//!     Mutable reference to topology. Traversal is limited, only yielding
//!     `Orphan...Mut` types. Allows both topology and geometry to mutate.
//!   - `Orphan...Mut`
//!     Mutable reference to geometry paired with a topological key. Cannot be
//!     traversed, but can mutate geometry and be collected.
//!
//! Note that it is not possible to get mutable views from another mutable
//! view, because a mutation may alter the topology and invalidate the
//! originating view. This means that mutable operations will always consume
//! `self` and that it is not possible to get a `...Mut` type from another
//! `...Mut` type. In general, an immutable traversal of topology can be used
//! to collect keys that are later used to query and mutate the target
//! topology.

// This code assumes that any keys for topological structures in the mesh are
// valid (hence the `unwrap` calls), which is very important for `Deref`.
// Topological mutations using views like `FaceView` are dangerous if they do
// not consume `self`. If these views can be used to mutate that data, then
// they can also invalidate these constraints and cause panics. Any mutating
// functions should consume the view.
//
// Similarly, toplogical mutations could invalidate views used to reach other
// views. This means that it is unsafe for a mutable view to yield another
// mutable view, because the second view may cause mutations that invalidate
// the first. Circulators effectively map from a mutable view to orphan views,
// for example. While `into` and `as` functions are okay, `as...mut` functions
// MUST yield orphans (or not exist at all).

use graph::Mesh;
use graph::geometry::Attribute;
use graph::storage::OpaqueKey;

mod edge;
mod face;
mod vertex;

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
