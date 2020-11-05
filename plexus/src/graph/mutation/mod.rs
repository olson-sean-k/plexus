pub mod edge;
pub mod face;
pub mod path;
pub mod vertex;

use fnv::FnvBuildHasher;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use crate::entity::storage::{AsStorage, AsStorageMut, Fuse, Journaled, StorageTarget};
use crate::entity::Entity;
use crate::graph::core::{Core, OwnedCore};
use crate::graph::data::{Data, Parametric};
use crate::graph::edge::{Arc, Edge};
use crate::graph::face::Face;
use crate::graph::mutation::face::FaceMutation;
use crate::graph::vertex::{Vertex, VertexKey};
use crate::graph::{GraphData, GraphError, GraphKey};
use crate::transact::{Bypass, Transact};

// TODO: The mutation API exposes raw entities (see removals). It would be ideal
//       if those types need not be exposed at all, since they have limited
//       utility to users. Is it possible to expose user data instead of
//       entities in these APIs?
// TODO: The stable toolchain does not allow a type parameter `G` to be
//       introduced and bound to the associated type `Mode::Graph::Data`. The
//       compiler does not seem to consider the types equal, and requires
//       redundant type bounds on `Mode`'s associated storage types at each
//       usage. The nightly toolchain already supports this. Reintroduce a
//       `G: GraphData` type parameter in implementation blocks when this is
//       fixed. For now, this code uses `Data<P::Graph>`. See the following
//       related issues:
//
//       https://github.com/rust-lang/rust/issues/58231
//       https://github.com/rust-lang/rust/issues/70703
//       https://github.com/rust-lang/rust/issues/47897

type StorageOf<E> = <E as Entity>::Storage;
type JournalOf<E> = Journaled<StorageOf<E>, E>;

type Rekeying = HashMap<GraphKey, GraphKey, FnvBuildHasher>;

/// Marker trait for graph representations that promise to be in a consistent
/// state.
///
/// This trait is only implemented by representations that ensure that their
/// storage is only ever mutated via the mutation API (and therefore is
/// consistent). Note that `Core` does not implement this trait and instead acts
/// as a raw container for topological storage that can be freely manipulated.
///
/// This trait allows code to make assumptions about the data it operates
/// against. For example, views expose an API to user code that assumes that
/// topologies are present and therefore unwraps values.
pub trait Consistent {}

impl<'a, T> Consistent for &'a T where T: Consistent {}

impl<'a, T> Consistent for &'a mut T where T: Consistent {}

pub trait Mode {
    type Graph: Parametric;
    type VertexStorage: AsStorageMut<Vertex<Data<Self::Graph>>>;
    type ArcStorage: AsStorageMut<Arc<Data<Self::Graph>>>;
    type EdgeStorage: AsStorageMut<Edge<Data<Self::Graph>>>;
    type FaceStorage: AsStorageMut<Face<Data<Self::Graph>>>;
}

pub struct Immediate<M>
where
    M: Parametric,
{
    phantom: PhantomData<M>,
}

impl<M> Mode for Immediate<M>
where
    M: Parametric,
{
    type Graph = M;
    type VertexStorage = StorageOf<Vertex<Data<M>>>;
    type ArcStorage = StorageOf<Arc<Data<M>>>;
    type EdgeStorage = StorageOf<Edge<Data<M>>>;
    type FaceStorage = StorageOf<Face<Data<M>>>;
}

pub struct Transacted<M>
where
    M: Parametric,
{
    phantom: PhantomData<M>,
}

impl<M> Mode for Transacted<M>
where
    M: Parametric,
{
    type Graph = M;
    type VertexStorage = JournalOf<Vertex<Data<M>>>;
    type ArcStorage = JournalOf<Arc<Data<M>>>;
    type EdgeStorage = JournalOf<Edge<Data<M>>>;
    type FaceStorage = JournalOf<Face<Data<M>>>;
}

/// Graph mutation.
pub struct Mutation<P>
where
    P: Mode,
    P::Graph: Consistent + From<OwnedCore<Data<P::Graph>>> + Into<OwnedCore<Data<P::Graph>>>,
{
    inner: FaceMutation<P>,
}

impl<P> AsRef<Self> for Mutation<P>
where
    P: Mode,
    P::Graph: Consistent + From<OwnedCore<Data<P::Graph>>> + Into<OwnedCore<Data<P::Graph>>>,
{
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<P> AsMut<Self> for Mutation<P>
where
    P: Mode,
    P::Graph: Consistent + From<OwnedCore<Data<P::Graph>>> + Into<OwnedCore<Data<P::Graph>>>,
{
    fn as_mut(&mut self) -> &mut Self {
        self
    }
}

impl<P> AsStorage<Arc<Data<P::Graph>>> for Mutation<P>
where
    P: Mode,
    P::Graph: Consistent + From<OwnedCore<Data<P::Graph>>> + Into<OwnedCore<Data<P::Graph>>>,
{
    fn as_storage(&self) -> &StorageTarget<Arc<Data<P::Graph>>> {
        self.inner.to_ref_core().unfuse().1
    }
}

impl<P> AsStorage<Edge<Data<P::Graph>>> for Mutation<P>
where
    P: Mode,
    P::Graph: Consistent + From<OwnedCore<Data<P::Graph>>> + Into<OwnedCore<Data<P::Graph>>>,
{
    fn as_storage(&self) -> &StorageTarget<Edge<Data<P::Graph>>> {
        self.inner.to_ref_core().unfuse().2
    }
}

impl<P> AsStorage<Face<Data<P::Graph>>> for Mutation<P>
where
    P: Mode,
    P::Graph: Consistent + From<OwnedCore<Data<P::Graph>>> + Into<OwnedCore<Data<P::Graph>>>,
{
    fn as_storage(&self) -> &StorageTarget<Face<Data<P::Graph>>> {
        self.inner.to_ref_core().unfuse().3
    }
}

impl<P> AsStorage<Vertex<Data<P::Graph>>> for Mutation<P>
where
    P: Mode,
    P::Graph: Consistent + From<OwnedCore<Data<P::Graph>>> + Into<OwnedCore<Data<P::Graph>>>,
{
    fn as_storage(&self) -> &StorageTarget<Vertex<Data<P::Graph>>> {
        self.inner.to_ref_core().unfuse().0
    }
}

impl<M> Bypass<M> for Mutation<Immediate<M>>
where
    M: Consistent + From<OwnedCore<Data<M>>> + Parametric + Into<OwnedCore<Data<M>>>,
{
    fn bypass(self) -> Self::Commit {
        self.inner.bypass().into()
    }
}

// TODO: This is a hack. Replace this with delegation.
impl<P> Deref for Mutation<P>
where
    P: Mode,
    P::Graph: Consistent + From<OwnedCore<Data<P::Graph>>> + Into<OwnedCore<Data<P::Graph>>>,
{
    type Target = FaceMutation<P>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<P> DerefMut for Mutation<P>
where
    P: Mode,
    P::Graph: Consistent + From<OwnedCore<Data<P::Graph>>> + Into<OwnedCore<Data<P::Graph>>>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

// TODO: The type bounds in this implementation are redundant but required. This
//       is probably a compiler bug and occurs because the type parameter `M` is
//       used as an alias for `P::Graph`. The aliasing is necessary to avoid
//       conflicts with the identity implementation of `From` in `core`, which
//       is also likely a compiler bug. See comments at the top of this module.
//impl<P, M> From<M> for Mutation<P>
//where
//    P: Mode<Graph = M>,
//    M: Consistent + From<OwnedCore<Data<M>>> + Parametric + Into<OwnedCore<Data<M>>>,
//    P::VertexStorage: AsStorageMut<Vertex<Data<M>>> + From<<Vertex<Data<M>> as Entity>::Storage>,
//    P::ArcStorage: AsStorageMut<Arc<Data<M>>> + From<<Arc<Data<M>> as Entity>::Storage>,
//    P::EdgeStorage: AsStorageMut<Edge<Data<M>>> + From<<Edge<Data<M>> as Entity>::Storage>,
//    P::FaceStorage: AsStorageMut<Face<Data<M>>> + From<<Face<Data<M>> as Entity>::Storage>,
//{
//    fn from(graph: M) -> Self {
//        Mutation {
//            inner: graph.into().into(),
//        }
//    }
//}

impl<M> From<M> for Mutation<Immediate<M>>
where
    M: Consistent + From<OwnedCore<Data<M>>> + Parametric + Into<OwnedCore<Data<M>>>,
{
    fn from(graph: M) -> Self {
        Mutation {
            inner: graph.into().into(),
        }
    }
}

impl<M> From<M> for Mutation<Transacted<M>>
where
    M: Consistent + From<OwnedCore<Data<M>>> + Parametric + Into<OwnedCore<Data<M>>>,
{
    fn from(graph: M) -> Self {
        let (vertices, arcs, edges, faces) = Into::<Core<_, _, _, _, _>>::into(graph).unfuse();
        let core = Core::empty()
            .fuse(Journaled::from(vertices))
            .fuse(Journaled::from(arcs))
            .fuse(Journaled::from(edges))
            .fuse(Journaled::from(faces));
        Mutation { inner: core.into() }
    }
}

impl<P> Parametric for Mutation<P>
where
    P: Mode,
    P::Graph: Consistent + From<OwnedCore<Data<P::Graph>>> + Into<OwnedCore<Data<P::Graph>>>,
{
    type Data = Data<P::Graph>;
}

impl<M> Transact<M> for Mutation<Immediate<M>>
where
    M: Consistent + From<OwnedCore<Data<M>>> + Parametric + Into<OwnedCore<Data<M>>>,
{
    type Commit = M;
    type Abort = ();
    type Error = GraphError;

    fn commit(self) -> Result<Self::Commit, (Self::Abort, Self::Error)> {
        self.inner.commit().map(|core| core.into())
    }

    fn abort(self) -> Self::Abort {}
}

impl<M> Transact<M> for Mutation<Transacted<M>>
where
    M: Consistent + From<OwnedCore<Data<M>>> + Parametric + Into<OwnedCore<Data<M>>>,
{
    type Commit = (M, Rekeying);
    type Abort = M;
    type Error = GraphError;

    fn commit(self) -> Result<Self::Commit, (Self::Abort, Self::Error)> {
        fn wrap<T, U>((from, to): (T, T)) -> (GraphKey, GraphKey)
        where
            T: Borrow<U>,
            U: Copy + Into<GraphKey>,
        {
            (from.borrow().clone().into(), to.borrow().clone().into())
        }
        self.inner
            .commit()
            .map(|core| {
                let mut rekeying = Rekeying::default();
                let (vertices, arcs, edges, faces) = core.unfuse();
                let (vertices, keys) = vertices.commit_and_rekey();
                rekeying.extend(keys.iter().map(wrap::<_, VertexKey>));
                let (arcs, keys) = arcs.commit_with_rekeying(&keys);
                rekeying.extend(keys.into_iter().map(wrap));
                let (edges, keys) = edges.commit_and_rekey();
                rekeying.extend(keys.into_iter().map(wrap));
                let (faces, keys) = faces.commit_and_rekey();
                rekeying.extend(keys.into_iter().map(wrap));
                (
                    Core::empty()
                        .fuse(vertices)
                        .fuse(arcs)
                        .fuse(edges)
                        .fuse(faces)
                        .into(),
                    rekeying,
                )
            })
            .map_err(|(core, error)| {
                let (vertices, arcs, edges, faces) = core.unfuse();
                (
                    Core::empty()
                        .fuse(vertices.abort())
                        .fuse(arcs.abort())
                        .fuse(edges.abort())
                        .fuse(faces.abort())
                        .into(),
                    error,
                )
            })
    }

    fn abort(self) -> Self::Abort {
        let (vertices, arcs, edges, faces) = self.inner.abort().unfuse();
        Core::empty()
            .fuse(vertices.abort())
            .fuse(arcs.abort())
            .fuse(edges.abort())
            .fuse(faces.abort())
            .into()
    }
}

pub trait Mutable:
    Consistent + From<OwnedCore<Data<Self>>> + Parametric + Into<OwnedCore<Data<Self>>>
{
}

impl<M, G> Mutable for M
where
    M: Consistent + From<OwnedCore<G>> + Parametric<Data = G> + Into<OwnedCore<G>>,
    G: GraphData,
{
}
