use fool::BoolExt;
use std::borrow::Borrow;
use std::collections::{HashSet, VecDeque};

use crate::entity::borrow::{Reborrow, ReborrowInto, ReborrowMut};
use crate::entity::storage::AsStorage;
use crate::entity::view::{Bind, ClosedView};
use crate::graph::edge::{Arc, ArcKey, ArcView};
use crate::graph::geometry::{Geometric, Geometry, GraphGeometry};
use crate::graph::mutation::Consistent;
use crate::graph::vertex::{Vertex, VertexKey, VertexView};
use crate::graph::{GraphError, OptionExt as _, Selector};
use crate::IteratorExt as _;

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
#[derive(Clone)]
pub struct Path<B>
where
    B: Reborrow,
    B::Target:
        AsStorage<Arc<Geometry<B>>> + AsStorage<Vertex<Geometry<B>>> + Consistent + Geometric,
{
    keys: VecDeque<ArcKey>,
    storage: B,
}

impl<B, M, G> Path<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
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
        let a = keys.next().ok_or_else(|| GraphError::TopologyMalformed)?;
        let b = keys.next().ok_or_else(|| GraphError::TopologyMalformed)?;
        let ab = (a, b).into();
        ArcView::bind(storage.reborrow(), ab).ok_or_else(|| GraphError::TopologyNotFound)?;
        let mut path = Path {
            keys: (&[ab]).iter().cloned().collect(),
            storage,
        };
        for key in keys {
            path.push_front(Selector::ByKey(key))?;
        }
        Ok(path)
    }

    pub fn to_ref(&self) -> Path<&M> {
        let storage = self.storage.reborrow();
        let keys = self.keys.iter();
        Path::bind_unchecked(storage, keys)
    }

    /// Pushes a vertex onto the back of the path.
    ///
    /// The back of a path $\overrightarrow{(A,\cdots)}$ is the vertex $A$.
    /// This is the source vertex of the first arc that forms the path.
    ///
    /// The given vertex must be a source vertex of an arc formed with the the
    /// back of the path. That is, if the given vertex is $X$, then
    /// $\overrightarrow{XA}$ must exist.
    ///
    /// Returns the key of the arc $\overrightarrow{XA}$ inserted into the path
    /// using the given source vertex $X$.
    ///
    /// # Errors
    ///
    /// Returns an error if the path is closed, the given vertex is not found,
    /// or the given vertex does not form an arc with the back of the path.
    pub fn push_back(&mut self, destination: Selector<VertexKey>) -> Result<ArcKey, GraphError> {
        self.is_open()
            .ok_or_else(|| GraphError::TopologyMalformed)?;
        let back = self.back();
        let xa = match destination {
            Selector::ByKey(key) => back
                .incoming_arcs()
                .find(|arc| arc.into_source_vertex().key() == key)
                .ok_or_else(|| GraphError::TopologyMalformed)?
                .key(),
            Selector::ByIndex(index) => {
                let x = back
                    .adjacent_vertices()
                    .keys()
                    .nth(index)
                    .ok_or_else(|| GraphError::TopologyNotFound)?;
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
            self.keys.push_back(xa);
            Ok(xa)
        }
    }

    /// Pops a vertex from the back of the path.
    pub fn pop_back(&mut self) -> Option<ArcKey> {
        // Empty paths are forbidden.
        if self.keys.len() > 1 {
            self.keys.pop_back()
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
    /// the front of the path. That is, if the given vertex is $X$, then
    /// $\overrightarrow{BX}$ must exist.
    ///
    /// Returns the key of the arc $\overrightarrow{BX}$ inserted into the path
    /// using the given source vertex $X$.
    ///
    /// # Errors
    ///
    /// Returns an error if the path is closed, the given vertex is not found,
    /// or the given vertex does not form an arc with the front of the path.
    pub fn push_front(&mut self, destination: Selector<VertexKey>) -> Result<ArcKey, GraphError> {
        self.is_open()
            .ok_or_else(|| GraphError::TopologyMalformed)?;
        let front = self.front();
        let bx = match destination {
            Selector::ByKey(key) => front
                .outgoing_arcs()
                .find(|arc| arc.into_destination_vertex().key() == key)
                .ok_or_else(|| GraphError::TopologyMalformed)?
                .key(),
            Selector::ByIndex(index) => {
                let x = front
                    .adjacent_vertices()
                    .keys()
                    .nth(index)
                    .ok_or_else(|| GraphError::TopologyNotFound)?;
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
            self.keys.push_front(bx);
            Ok(bx)
        }
    }

    /// Pops a vertex from the front of the path.
    pub fn pop_front(&mut self) -> Option<ArcKey> {
        // Empty paths are forbidden.
        if self.keys.len() > 1 {
            self.keys.pop_front()
        }
        else {
            None
        }
    }

    /// Gets the vertex at the back of the path.
    pub fn back(&self) -> VertexView<&M> {
        let (key, _) = self.endpoints();
        Bind::bind(self.storage.reborrow(), key).expect_consistent()
    }

    /// Gets the vertex at the front of the path.
    pub fn front(&self) -> VertexView<&M> {
        let (_, key) = self.endpoints();
        Bind::bind(self.storage.reborrow(), key).expect_consistent()
    }

    /// Gets an iterator over the vertices in the path.
    pub fn vertices<'a>(&'a self) -> impl Iterator<Item = VertexView<&'a M>>
    where
        M: 'a,
    {
        let back = self.back();
        Some(back)
            .into_iter()
            .chain(self.arcs().map(|arc| arc.into_destination_vertex()))
    }

    /// Gets an iterator over the arcs in the path.
    pub fn arcs<'a>(&'a self) -> impl ExactSizeIterator<Item = ArcView<&'a M>>
    where
        M: 'a,
    {
        let storage = self.storage.reborrow();
        self.keys
            .iter()
            .rev()
            .cloned()
            .map(move |key| Bind::bind(storage, key).expect_consistent())
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
        let (a, b) = self.endpoints();
        a == b
    }

    fn bind_unchecked<I>(storage: B, keys: I) -> Self
    where
        I: IntoIterator,
        I::Item: Borrow<ArcKey>,
    {
        let keys = keys.into_iter().map(|key| *key.borrow()).collect();
        Path { storage, keys }
    }

    fn endpoints(&self) -> (VertexKey, VertexKey) {
        let (a, _) = self.keys.back().cloned().expect("empty path").into();
        let (_, b) = self.keys.front().cloned().expect("empty pathy").into();
        (a, b)
    }
}

impl<B, M, G> Path<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    pub fn to_mut(&mut self) -> Path<&mut M> {
        let storage = self.storage.reborrow_mut();
        let keys = self.keys.iter();
        Path::bind_unchecked(storage, keys)
    }
}

impl<'a, B, M, G> Path<B>
where
    B: ReborrowInto<'a, Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    /// Converts a mutable view into an immutable view.
    ///
    /// This is useful when mutations are not (or no longer) needed and mutual
    /// access is desired.
    pub fn into_ref(self) -> Path<&'a M> {
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
    /// `ArcView::split` or `FaceView::split`, for example). To split a graph
    /// along a path (and thus mutate the graph) use `MeshGraph::split_at_path`.
    ///
    /// It is not possible to split a path at its back or front vertices.
    ///
    /// # Errors
    ///
    /// Returns an error if the given vertex cannot be found or the path cannot
    /// be split at that vertex.
    pub fn split(self, at: Selector<VertexKey>) -> Result<(Path<&'a M>, Path<&'a M>), GraphError> {
        let index = at.index_or_else(|key| {
            self.vertices()
                .keys()
                .enumerate()
                .find(|(_, a)| *a == key)
                .map(|(n, _)| n)
                .ok_or_else(|| GraphError::TopologyNotFound)
        })?;
        if index == 0 || index >= self.keys.len() {
            return Err(GraphError::TopologyMalformed);
        }
        let Path {
            keys: mut right,
            storage,
            ..
        } = self.into_ref();
        let left = right.split_off(index);
        Ok((
            Path::bind_unchecked(storage, left),
            Path::bind_unchecked(storage, right),
        ))
    }
}

impl<B, M, G> PartialEq for Path<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn eq(&self, other: &Self) -> bool {
        let keys = |path: &Self| path.arcs().keys().collect::<HashSet<_>>();
        keys(self) == keys(other)
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Point2;

    use crate::graph::{ClosedView, MeshGraph, Selector};
    use crate::prelude::*;
    use crate::primitive::{Tetragon, Trigon};
    use crate::IteratorExt;

    use Selector::ByKey;

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
            .interior_arcs()
            .map(|arc| arc.into_source_vertex())
            .keys()
            .collect::<Vec<_>>();

        let mut path = graph.path(keys.iter()).unwrap();
        assert!(path.is_open());
        // TODO: Move this assertion to a distinct test.
        assert_eq!(path.vertices().keys().collect::<Vec<_>>(), keys.to_vec());

        path.push_front(ByKey(keys[0])).unwrap();
        assert!(path.is_closed());
        assert_eq!(path.front().key(), path.back().key());
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

        let (left, right) = path.split(ByIndex(1)).unwrap();
        assert_eq!(left.front().key(), right.back().key());
        assert_eq!(left.arcs().count(), 1);
        assert_eq!(right.arcs().count(), 1);
    }
}
