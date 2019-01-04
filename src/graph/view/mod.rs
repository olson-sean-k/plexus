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

pub mod convert;
mod edge;
mod face;
mod vertex;

pub use self::edge::{EdgeKeyTopology, EdgeView, OrphanEdgeView};
pub use self::face::{FaceKeyTopology, FaceView, OrphanFaceView, RegionView};
pub use self::vertex::{OrphanVertexView, VertexView};
