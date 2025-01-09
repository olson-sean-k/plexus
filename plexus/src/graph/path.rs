use itertools::Itertools;
use std::borrow::{Borrow, Cow};
use std::cmp;
use std::collections::{HashSet, VecDeque};

use crate::entity::borrow::{Reborrow, ReborrowInto};
use crate::entity::storage::AsStorage;
use crate::entity::view::{Bind, ClosedView, Unbind, View};
use crate::geometry::Metric;
use crate::graph::data::{Data, GraphData, Parametric};
use crate::graph::edge::{Arc, ArcKey, ArcView, Edge};
use crate::graph::face::{Face, FaceView, Ring};
use crate::graph::mutation::path::{self, PathExtrudeCache};
use crate::graph::mutation::{self, Consistent, Immediate, Mutable};
use crate::graph::vertex::{Vertex, VertexKey, VertexView};
use crate::graph::{GraphError, OptionExt as _, ResultExt as _, Selector};
use crate::transact::{BypassOrCommit, Mutate};
use crate::IteratorExt as _;

type Mutation<M> = mutation::Mutation<Immediate<M>>;

/// Non-intersecting path.
///
/// A path is an ordered set of vertices that are joined by arcs. Paths are
/// notated as either sequences or sets. An open path over vertices $A$, $B$,
/// and $C$ is notated $\overrightarrow{(A,B,C)}$ and a closed path over the
/// same vertices is notated $\overrightarrow{\\{A,B,C\\}}$.
///
/// `Path` represents non-intersecting paths of the form
/// $\overrightarrow{(A,\cdots,B)}$, where $A$ is the _back_ of the path and $B$
/// is the _front_ of the path. Note that closed paths are always of the form
/// $\overrightarrow{(A,\cdots,A)}$, where the back and front vertices are both
/// $A$ (the same).
///
/// `Path` maintains an ordered set of keys and uses copy-on-write semantics to
/// avoid allocations and copies.
#[derive(Clone)]
pub struct Path<'k, B>
where
    B: Reborrow,
    B::Target: AsStorage<Arc<Data<B>>> + AsStorage<Vertex<Data<B>>> + Consistent + Parametric,
{
    keys: Cow<'k, VecDeque<ArcKey>>,
    storage: B,
}

impl<B, M, G> Path<'static, B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent + Parametric<Data = G>,
    G: GraphData,
{
    // Paths bind multiple keys to storage and so do not support view APIs.
    // This bespoke `bind` function ensures that the path is not empty and that
    // the topology forms a non-intersecting path.
    pub(in crate::graph) fn bind<I>(storage: B, keys: I) -> Result<Self, GraphError>
    where
        I: IntoIterator,
        I::Item: Borrow<VertexKey>,
    {
        let mut keys = keys.into_iter().map(|key| *key.borrow());
        let a = keys.next().ok_or(GraphError::TopologyMalformed)?;
        let b = keys.next().ok_or(GraphError::TopologyMalformed)?;
        let ab = (a, b).into();
        ArcView::bind(storage.reborrow(), ab).ok_or(GraphError::TopologyNotFound)?;
        let mut path = Path {
            keys: Cow::Owned([ab].into_iter().collect()),
            storage,
        };
        for key in keys {
            path.push_front(key)?;
        }
        Ok(path)
    }

    fn bind_unchecked<I>(storage: B, keys: I) -> Self
    where
        I: IntoIterator,
        I::Item: Borrow<ArcKey>,
    {
        let keys = Cow::Owned(keys.into_iter().map(|key| *key.borrow()).collect());
        Path { keys, storage }
    }
}

impl<B, M, G> Path<'_, B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent + Parametric<Data = G>,
    G: GraphData,
{
    pub fn to_ref(&self) -> Path<&M> {
        Path {
            storage: self.storage.reborrow(),
            keys: Cow::Borrowed(&self.keys),
        }
    }

    /// Converts the path into its opposite path.
    pub fn into_opposite_path(self) -> Path<'static, B> {
        let Path { keys, storage } = self;
        Path::bind_unchecked(
            storage,
            keys.iter().cloned().rev().map(|ab| ab.into_opposite()),
        )
    }

    /// Gets the opposite path.
    pub fn opposite_path(&self) -> Path<'static, &M> {
        self.to_ref().into_opposite_path()
    }

    /// Pushes a vertex onto the back of the path.
    ///
    /// The back of a path $\overrightarrow{(A,\cdots)}$ is the vertex $A$.
    /// This is the source vertex of the first arc that forms the path.
    ///
    /// The given vertex must be a source vertex of an arc formed with the back
    /// of the path. That is, if the given vertex is $X$, then
    /// $\overrightarrow{XA}$ must exist.
    ///
    /// Returns the key of the arc $\overrightarrow{XA}$ inserted into the path
    /// using the given source vertex $X$.
    ///
    /// # Errors
    ///
    /// Returns an error if the path is closed, the given vertex is not found,
    /// or the given vertex does not form an arc with the back of the path.
    pub fn push_back(
        &mut self,
        destination: impl Into<Selector<VertexKey>>,
    ) -> Result<ArcKey, GraphError> {
        if self.is_closed() {
            return Err(GraphError::TopologyMalformed);
        }
        let back = self.back();
        let xa = match destination.into() {
            Selector::ByKey(key) => back
                .incoming_arcs()
                .find(|arc| arc.into_source_vertex().key() == key)
                .ok_or(GraphError::TopologyMalformed)?
                .key(),
            Selector::ByIndex(index) => {
                let x = back
                    .adjacent_vertices()
                    .keys()
                    .nth(index)
                    .ok_or(GraphError::TopologyNotFound)?;
                (x, back.key()).into()
            }
        };
        let (x, _) = xa.into();
        // Do not allow intersections unless they form a loop with the first
        // vertex in the path (this iteration skips the vertex at the front of
        // the path).
        let is_intersecting = self
            .arcs()
            .map(|arc| arc.into_source_vertex())
            .keys()
            .any(|key| key == x);
        if is_intersecting {
            Err(GraphError::TopologyMalformed)
        }
        else {
            self.keys.to_mut().push_back(xa);
            Ok(xa)
        }
    }

    /// Pushes the source vertex of the previous arc onto the back of the path.
    pub fn push_previous_arc(&mut self) -> Result<ArcKey, GraphError> {
        let key = *self.keys.to_mut().back().expect("empty path");
        let key = ArcView::from(View::bind_unchecked(self.storage.reborrow(), key))
            .into_previous_arc()
            .into_source_vertex()
            .key();
        self.push_back(key)
    }

    /// Pops a vertex from the back of the path.
    pub fn pop_back(&mut self) -> Option<ArcKey> {
        // Empty paths are forbidden.
        if self.keys.len() > 1 {
            self.keys.to_mut().pop_back()
        }
        else {
            None
        }
    }

    /// Pushes a vertex onto the front of the path.
    ///
    /// The front of a path $\overrightarrow{(\cdots,B)}$ is the vertex $B$.
    /// This is the destination vertex of the last arc that forms the path.
    ///
    /// The given vertex must be a destination vertex of an arc formed with the
    /// front of the path.  That is, if the given vertex is $X$, then
    /// $\overrightarrow{BX}$ must exist.
    ///
    /// Returns the key of the arc $\overrightarrow{BX}$ inserted into the path
    /// using the given source vertex $X$.
    ///
    /// # Errors
    ///
    /// Returns an error if the path is closed, the given vertex is not found,
    /// or the given vertex does not form an arc with the front of the path.
    pub fn push_front(
        &mut self,
        destination: impl Into<Selector<VertexKey>>,
    ) -> Result<ArcKey, GraphError> {
        if self.is_closed() {
            return Err(GraphError::TopologyMalformed);
        }
        let front = self.front();
        let bx = match destination.into() {
            Selector::ByKey(key) => front
                .outgoing_arcs()
                .find(|arc| arc.into_destination_vertex().key() == key)
                .ok_or(GraphError::TopologyMalformed)?
                .key(),
            Selector::ByIndex(index) => {
                let x = front
                    .adjacent_vertices()
                    .keys()
                    .nth(index)
                    .ok_or(GraphError::TopologyNotFound)?;
                (front.key(), x).into()
            }
        };
        let (_, x) = bx.into();
        // Do not allow intersections unless they form a loop with the first
        // vertex in the path (this iteration skips the vertex at the back of
        // the path).
        let is_intersecting = self
            .arcs()
            .map(|arc| arc.into_destination_vertex())
            .keys()
            .any(|key| key == x);
        if is_intersecting {
            Err(GraphError::TopologyMalformed)
        }
        else {
            self.keys.to_mut().push_front(bx);
            Ok(bx)
        }
    }

    /// Pushes the destination vertex of the next arc onto the front of the
    /// path.
    pub fn push_next_arc(&mut self) -> Result<ArcKey, GraphError> {
        let key = *self.keys.front().expect("empty path");
        let key = ArcView::from(View::bind_unchecked(self.storage.reborrow(), key))
            .into_next_arc()
            .into_destination_vertex()
            .key();
        self.push_front(key)
    }

    /// Pops a vertex from the front of the path.
    pub fn pop_front(&mut self) -> Option<ArcKey> {
        // Empty paths are forbidden.
        if self.keys.len() > 1 {
            self.keys.to_mut().pop_front()
        }
        else {
            None
        }
    }

    /// Gets the vertex at the back of the path.
    pub fn back(&self) -> VertexView<&M> {
        let (key, _) = self.terminals();
        View::<_, Vertex<_>>::bind_unchecked(self.storage.reborrow(), key).into()
    }

    /// Gets the vertex at the front of the path.
    pub fn front(&self) -> VertexView<&M> {
        let (_, key) = self.terminals();
        View::<_, Vertex<_>>::bind_unchecked(self.storage.reborrow(), key).into()
    }

    pub fn shortest_metric_with<Q, F>(
        &self,
        from: impl Into<Selector<VertexKey>>,
        to: impl Into<Selector<VertexKey>>,
        f: F,
    ) -> Result<Q, GraphError>
    where
        Q: Metric,
        F: Fn(VertexView<&M>, VertexView<&M>) -> Q,
    {
        self.shortest_subpath_terminals(from, to).map(|(from, to)| {
            // A cycle is needed for closed paths. Note that if the path is
            // open, then the vertex keys must not wrap over the terminals here.
            truncate(self.arcs().cycle(), from, to).fold(Q::zero(), |metric, arc| {
                metric + f(arc.source_vertex(), arc.destination_vertex())
            })
        })
    }

    /// Returns `true` if the path is open.
    ///
    /// An _open path_ is a path that terminates and does **not** form a loop.
    pub fn is_open(&self) -> bool {
        !self.is_closed()
    }

    /// Returns `true` if the path is closed.
    ///
    /// A _closed path_ is a path that forms a loop by starting and ending at
    /// the same vertex.
    pub fn is_closed(&self) -> bool {
        let (a, b) = self.terminals();
        a == b
    }

    fn terminals(&self) -> (VertexKey, VertexKey) {
        let (a, _) = self.keys.back().cloned().expect("empty path").into();
        let (_, b) = self.keys.front().cloned().expect("empty path").into();
        (a, b)
    }

    fn shortest_subpath_terminals(
        &self,
        from: impl Into<Selector<VertexKey>>,
        to: impl Into<Selector<VertexKey>>,
    ) -> Result<(VertexKey, VertexKey), GraphError> {
        let index_key = |selector| {
            match selector {
                Selector::ByKey(key) => self
                    .vertices()
                    .find_position(|vertex| vertex.key() == key)
                    .map(|(index, _)| (index, key)),
                Selector::ByIndex(index) => self
                    .vertices()
                    .nth(index)
                    .map(|vertex| (index, vertex.key())),
            }
            .ok_or(GraphError::TopologyNotFound)
        };
        let (i, from) = index_key(from.into())?;
        let (j, to) = index_key(to.into())?;
        if self.is_open() {
            // Reorder the vertex keys if they oppose the direction of the open
            // path.
            Ok(if i > j { (to, from) } else { (from, to) })
        }
        else {
            // Reorder the vertex keys if they form the longer closed path.
            let n = self.keys.len() / 2;
            Ok(if (cmp::max(i, j) - cmp::min(i, j)) > n {
                (to, from)
            }
            else {
                (from, to)
            })
        }
    }
}

impl<'k, 'a, B, M, G> Path<'k, B>
where
    B: ReborrowInto<'a, Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent + Parametric<Data = G>,
    G: GraphData,
{
    /// Converts a mutable view into an immutable view.
    pub fn into_ref(self) -> Path<'k, &'a M> {
        let Path { keys, storage, .. } = self;
        Path {
            keys,
            storage: storage.reborrow_into(),
        }
    }

    /// Splits the path into two immutable paths at the given vertex.
    ///
    /// Given a path $\overrightarrow{(A,\cdots,M,\cdots,B)}$, splitting at the
    /// vertex $M$ results in the paths $\overrightarrow{(A,\cdots,M)}$ and
    /// $\overrightarrow{(M,\cdots,B)}$.
    ///
    /// **Splitting a path does not mutate its graph in any way** (unlike
    /// [`ArcView::split_with`] or [`FaceView::split`], for example). To split a
    /// graph along a path (and thus mutate the graph) use
    /// [`MeshGraph::split_at_path`].
    ///
    /// It is not possible to split a path at its back or front vertices.
    ///
    /// # Errors
    ///
    /// Returns an error if the given vertex cannot be found or the path cannot
    /// be split at that vertex.
    ///
    /// [`ArcView::split_with`]: crate::graph::ArcView::split_with
    /// [`FaceView::split`]: crate::graph::FaceView::split
    /// [`MeshGraph::split_at_path`]: crate::graph::MeshGraph::split_at_path
    pub fn split(
        self,
        at: impl Into<Selector<VertexKey>>,
    ) -> Result<(Path<'static, &'a M>, Path<'static, &'a M>), GraphError> {
        let index = at.into().index_or_else(|key| {
            self.vertices()
                .keys()
                .enumerate()
                .find(|(_, a)| *a == key)
                .map(|(n, _)| n)
                .ok_or(GraphError::TopologyNotFound)
        })?;
        if index == 0 || index >= self.keys.len() {
            return Err(GraphError::TopologyMalformed);
        }
        let Path { keys, storage, .. } = self.into_ref();
        let mut right = keys.into_owned();
        let left = right.split_off(index);
        Ok((
            Path {
                keys: Cow::Owned(left),
                storage,
            },
            Path {
                keys: Cow::Owned(right),
                storage,
            },
        ))
    }
}

impl<B, G> Path<'_, B>
where
    B: Reborrow,
    B::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent + Parametric<Data = G>,
    G: GraphData,
{
    /// Gets an iterator over the vertices in the path.
    pub fn vertices(&self) -> impl Clone + Iterator<Item = VertexView<&B::Target>> {
        let (key, _) = self.terminals();
        Some(key)
            .into_iter()
            .chain(self.keys.iter().cloned().rev().map(|key| {
                let (_, key) = key.into();
                key
            }))
            .map(move |key| View::bind_unchecked(self.storage.reborrow(), key))
            .map(From::from)
    }

    /// Gets an iterator over the arcs in the path.
    pub fn arcs(&self) -> impl Clone + ExactSizeIterator<Item = ArcView<&B::Target>> {
        self.keys
            .iter()
            .rev()
            .cloned()
            .map(move |key| View::bind_unchecked(self.storage.reborrow(), key))
            .map(From::from)
    }
}

impl<'a, M, G> Path<'_, &'a mut M>
where
    M: AsStorage<Arc<G>>
        + AsStorage<Edge<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + Default
        + Mutable<Data = G>,
    G: GraphData,
{
    /// Extrudes the contour of a boundary path.
    ///
    /// A path is a boundary path if all of its arcs are boundary arcs.
    /// Extruding the path transforms the vertices along the path in order using
    /// the given function and inserts a face between the path and its extruded
    /// contour.
    ///
    /// Unlike extruding individual arcs, extruding the contour of a path
    /// inserts a single face in which all involved arcs participate.
    ///
    /// Returns the inserted face.
    ///
    /// # Errors
    ///
    /// Returns an error if the path is not a boundary path.
    pub fn extrude_contour_with<F>(self, f: F) -> Result<FaceView<&'a mut M>, GraphError>
    where
        F: Fn(&G::Vertex) -> G::Vertex,
    {
        let cache = PathExtrudeCache::from_path(self.to_ref())?;
        let Path { storage, .. } = self;
        Ok(Mutation::take(storage)
            .bypass_or_commit_with(|mutation| path::extrude_contour_with(mutation, cache, f))
            .map(|(storage, face)| Bind::bind(storage, face).expect_consistent())
            .map_err(|(_, error)| error)
            .expect_consistent())
    }

    /// Extrudes the surface of a closed path.
    pub fn extrude_surface_with<F>(self, f: F) -> Result<Self, GraphError>
    where
        F: Fn(G::Vertex) -> G::Vertex,
    {
        let _ = f;
        todo!()
    }
}

impl<B, M, G> From<Ring<B>> for Path<'static, B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent + Parametric<Data = G>,
    G: GraphData,
{
    fn from(ring: Ring<B>) -> Self {
        let keys = Cow::Owned(ring.arcs().keys().collect());
        let (storage, _) = ring.into_arc().unbind();
        Path { keys, storage }
    }
}

impl<B, M, G> PartialEq for Path<'_, B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent + Parametric<Data = G>,
    G: GraphData,
{
    fn eq(&self, other: &Self) -> bool {
        let keys = |path: &Self| path.keys.iter().cloned().collect::<HashSet<_>>();
        keys(self) == keys(other)
    }
}

fn truncate<T>(
    arcs: impl IntoIterator<Item = T>,
    from: VertexKey,
    to: VertexKey,
) -> impl Iterator<Item = T>
where
    T: Borrow<ArcKey> + Copy,
{
    arcs.into_iter()
        .map(|arc| (arc, (*arc.borrow()).into()))
        .skip_while(move |(_, (a, _))| *a != from)
        .take_while(move |(_, (a, _))| *a != to)
        .map(|(arc, _)| arc)
}

#[cfg(test)]
mod tests {
    use nalgebra::Point2;

    use crate::graph::{ClosedView, MeshGraph};
    use crate::prelude::*;
    use crate::primitive::{Tetragon, Trigon};
    use crate::IteratorExt;

    type E2 = Point2<f64>;

    #[test]
    fn open_close() {
        let graph = MeshGraph::<E2>::from_raw_buffers(
            vec![Trigon::from([0usize, 1, 2])],
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        )
        .unwrap();
        let keys = graph
            .faces()
            .nth(0)
            .unwrap()
            .adjacent_arcs()
            .map(|arc| arc.into_source_vertex())
            .keys()
            .collect::<Vec<_>>();

        let mut path = graph.path(keys.iter()).unwrap();
        assert!(path.is_open());
        // TODO: Move this assertion to a distinct test.
        assert_eq!(path.vertices().keys().collect::<Vec<_>>(), keys.to_vec());

        path.push_front(keys[0]).unwrap();
        assert!(path.is_closed());
        assert_eq!(path.front().key(), path.back().key());
    }

    #[test]
    fn logical_metrics() {
        let graph = MeshGraph::<E2>::from_raw_buffers(
            vec![Tetragon::from([0usize, 1, 2, 3])],
            vec![(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)],
        )
        .unwrap();
        let path = {
            let key = graph.faces().nth(0).unwrap().key();
            graph.face(key).unwrap().into_ring().into_path()
        };
        let keys: Vec<_> = path.vertices().keys().collect();

        assert_eq!(0, path.shortest_metric_with(0, 0, |_, _| 1usize).unwrap());
        assert_eq!(
            0,
            path.shortest_metric_with(keys[0], keys[0], |_, _| 1usize)
                .unwrap()
        );
        assert_eq!(1, path.shortest_metric_with(0, 3, |_, _| 1usize).unwrap());
        assert_eq!(
            2,
            path.shortest_metric_with(keys[3], keys[1], |_, _| 1usize)
                .unwrap()
        );
    }

    #[test]
    fn split() {
        let graph =
            MeshGraph::<()>::from_raw_buffers(vec![Tetragon::from([0usize, 1, 2, 3])], vec![(); 4])
                .unwrap();
        let source = graph.vertices().nth(0).unwrap();
        let destination = source
            .into_outgoing_arc()
            .into_next_arc()
            .into_destination_vertex();

        let path = source.shortest_path(destination.key()).unwrap();
        assert_eq!(path.arcs().count(), 2);

        let (left, right) = path.split(1).unwrap();
        assert_eq!(left.front().key(), right.back().key());
        assert_eq!(left.arcs().count(), 1);
        assert_eq!(right.arcs().count(), 1);
    }
}
