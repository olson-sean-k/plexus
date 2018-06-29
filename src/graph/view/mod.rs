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

// TODO: The use of `from_keyed_storage` is dangerous! It cannot encode
//       consistency, and relies entirely on the caller to specify. Either it
//       should always yield inconsistent views (requiring the use of traits
//       whenever a consistent view is needed) or only the conversion traits
//       (and their required constraints) should be used.

#![macro_use]

macro_rules! interior_deref {
    (vertex => $v:expr) => {{
        let (key, storage) = $v.into_keyed_storage();
        VertexView::<_, _, Consistent>::from_keyed_storage(key, AsRef::<Mesh<G>>::as_ref(storage))
            .unwrap()
    }};
    (edge => $v:expr) => {{
        let (key, storage) = $v.into_keyed_storage();
        EdgeView::<_, _, Consistent>::from_keyed_storage(key, AsRef::<Mesh<G>>::as_ref(storage))
            .unwrap()
    }};
    (face => $v:expr) => {{
        let (key, storage) = $v.into_keyed_storage();
        FaceView::<_, _, Consistent>::from_keyed_storage(key, AsRef::<Mesh<G>>::as_ref(storage))
            .unwrap()
    }};
}

use std::marker::PhantomData;

use graph::mesh::Mesh;

pub mod convert;
mod edge;
mod face;
mod vertex;

pub use self::edge::{EdgeKeyTopology, EdgeView, OrphanEdgeView};
pub use self::face::{FaceKeyTopology, FaceView, OrphanFaceView};
pub use self::vertex::{OrphanVertexView, VertexView};

pub type EdgeRef<'a, G> = EdgeView<&'a Mesh<G>, G, Consistent>;
pub type EdgeMut<'a, G> = EdgeView<&'a mut Mesh<G>, G, Consistent>;
pub type OrphanEdge<'a, G> = OrphanEdgeView<'a, G>;

pub type FaceRef<'a, G> = FaceView<&'a Mesh<G>, G, Consistent>;
pub type FaceMut<'a, G> = FaceView<&'a mut Mesh<G>, G, Consistent>;
pub type OrphanFace<'a, G> = OrphanFaceView<'a, G>;

pub type VertexRef<'a, G> = VertexView<&'a Mesh<G>, G, Consistent>;
pub type VertexMut<'a, G> = VertexView<&'a mut Mesh<G>, G, Consistent>;
pub type OrphanVertex<'a, G> = OrphanVertexView<'a, G>;

pub trait Consistency {}

pub struct Consistent;

impl Consistency for Consistent {}

pub struct Inconsistent;

impl Consistency for Inconsistent {}

trait IteratorExt: Iterator + Sized {
    fn map_with_ref<F, R>(self, f: F) -> MapWithRef<Self, F, R>
    where
        F: FnMut(&Self, Self::Item) -> R,
    {
        MapWithRef {
            input: self,
            f,
            phantom: PhantomData,
        }
    }

    fn map_with_mut<F, R>(self, f: F) -> MapWithMut<Self, F, R>
    where
        F: FnMut(&mut Self, Self::Item) -> R,
    {
        MapWithMut {
            input: self,
            f,
            phantom: PhantomData,
        }
    }
}

impl<I> IteratorExt for I where I: Iterator + Sized {}

struct MapWithRef<I, F, R>
where
    I: Iterator,
    F: FnMut(&I, I::Item) -> R,
{
    input: I,
    f: F,
    phantom: PhantomData<R>,
}

impl<I, F, R> Iterator for MapWithRef<I, F, R>
where
    I: Iterator,
    F: FnMut(&I, I::Item) -> R,
{
    type Item = R;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.input.next();
        item.map(|item| (self.f)(&self.input, item))
    }
}

struct MapWithMut<I, F, R>
where
    I: Iterator,
    F: FnMut(&mut I, I::Item) -> R,
{
    input: I,
    f: F,
    phantom: PhantomData<R>,
}

impl<I, F, R> Iterator for MapWithMut<I, F, R>
where
    I: Iterator,
    F: FnMut(&mut I, I::Item) -> R,
{
    type Item = R;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.input.next();
        item.map(|item| (self.f)(&mut self.input, item))
    }
}
