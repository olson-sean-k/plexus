use smallvec::SmallVec;
use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::ops::{Deref, DerefMut};

use crate::entity::borrow::Reborrow;
use crate::entity::storage::{AsStorage, AsStorageMut, Fuse, StorageObject};
use crate::entity::view::{Bind, ClosedView, Rebind, Unbind};
use crate::graph::core::Core;
use crate::graph::data::{Data, GraphData, Parametric};
use crate::graph::edge::{Arc, ArcKey, ArcView, Edge};
use crate::graph::face::{Face, FaceKey, FaceView, ToRing};
use crate::graph::mutation::edge::{self, ArcBridgeCache, EdgeMutation};
use crate::graph::mutation::{vertex, Consistent, Immediate, Mode, Mutable, Mutation, Transacted};
use crate::graph::vertex::{Vertex, VertexKey, VertexView};
use crate::graph::GraphError;
use crate::transact::{Bypass, Transact};
use crate::{DynamicArity, IteratorExt as _};

type ModalCore<P> = Core<
    Data<<P as Mode>::Graph>,
    <P as Mode>::VertexStorage,
    <P as Mode>::ArcStorage,
    <P as Mode>::EdgeStorage,
    <P as Mode>::FaceStorage,
>;
#[cfg(not(all(nightly, feature = "unstable")))]
pub type RefCore<'a, G> = Core<
    G,
    &'a StorageObject<Vertex<G>>,
    &'a StorageObject<Arc<G>>,
    &'a StorageObject<Edge<G>>,
    &'a StorageObject<Face<G>>,
>;
#[cfg(all(nightly, feature = "unstable"))]
pub type RefCore<'a, G> = Core<
    G,
    &'a StorageObject<'a, Vertex<G>>,
    &'a StorageObject<'a, Arc<G>>,
    &'a StorageObject<'a, Edge<G>>,
    &'a StorageObject<'a, Face<G>>,
>;

pub struct FaceMutation<P>
where
    P: Mode,
{
    inner: EdgeMutation<P>,
    storage: P::FaceStorage,
}

impl<P> FaceMutation<P>
where
    P: Mode,
{
    pub fn to_ref_core(&self) -> RefCore<Data<P::Graph>> {
        self.inner.to_ref_core().fuse(self.storage.as_storage())
    }

    // TODO: Should there be a distinction between `connect_face_to_arc` and
    //       `connect_arc_to_face`?
    pub fn connect_face_to_arc(&mut self, ab: ArcKey, abc: FaceKey) -> Result<(), GraphError> {
        self.with_face_mut(abc, |face| face.arc = ab)
    }

    fn connect_face_interior(&mut self, arcs: &[ArcKey], face: FaceKey) -> Result<(), GraphError> {
        for (ab, bc) in arcs.iter().cloned().perimeter() {
            self.connect_adjacent_arcs(ab, bc)?;
            self.connect_arc_to_face(ab, face)?;
        }
        Ok(())
    }

    fn disconnect_face_interior(&mut self, arcs: &[ArcKey]) -> Result<(), GraphError> {
        for ab in arcs {
            self.disconnect_arc_from_face(*ab)?;
        }
        Ok(())
    }

    fn connect_face_exterior(
        &mut self,
        arcs: &[ArcKey],
        connectivity: (
            HashMap<VertexKey, Vec<ArcKey>>,
            HashMap<VertexKey, Vec<ArcKey>>,
        ),
    ) -> Result<(), GraphError> {
        let (incoming, outgoing) = connectivity;
        for ab in arcs.iter().cloned() {
            let (a, b) = ab.into();
            let ba = ab.into_opposite();
            let adjacent = {
                let core = &self.to_ref_core();
                if ArcView::bind(core, ba)
                    .ok_or_else(|| GraphError::TopologyMalformed)?
                    .is_boundary_arc()
                {
                    // The next arc of BA is the outgoing arc of the destination
                    // vertex A that is also a boundary arc or, if there is no
                    // such outgoing arc, the next exterior arc of the face. The
                    // previous arc is similar.
                    let ax = outgoing[&a]
                        .iter()
                        .cloned()
                        .flat_map(|ax| ArcView::bind(core, ax))
                        .find(|next| next.is_boundary_arc())
                        .or_else(|| {
                            ArcView::bind(core, ab)
                                .and_then(|arc| arc.into_reachable_previous_arc())
                                .and_then(|previous| previous.into_reachable_opposite_arc())
                        })
                        .map(|next| next.key());
                    let xb = incoming[&b]
                        .iter()
                        .cloned()
                        .flat_map(|xb| ArcView::bind(core, xb))
                        .find(|previous| previous.is_boundary_arc())
                        .or_else(|| {
                            ArcView::bind(core, ab)
                                .and_then(|arc| arc.into_reachable_next_arc())
                                .and_then(|next| next.into_reachable_opposite_arc())
                        })
                        .map(|previous| previous.key());
                    ax.into_iter().zip(xb.into_iter()).next()
                }
                else {
                    None
                }
            };
            if let Some((ax, xb)) = adjacent {
                self.connect_adjacent_arcs(ba, ax)?;
                self.connect_adjacent_arcs(xb, ba)?;
            }
        }
        Ok(())
    }

    fn with_face_mut<T, F>(&mut self, abc: FaceKey, mut f: F) -> Result<T, GraphError>
    where
        F: FnMut(&mut Face<Data<P::Graph>>) -> T,
    {
        let face = self
            .storage
            .as_storage_mut()
            .get_mut(&abc)
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        Ok(f(face))
    }
}

impl<P> AsStorage<Face<Data<P::Graph>>> for FaceMutation<P>
where
    P: Mode,
{
    fn as_storage(&self) -> &StorageObject<Face<Data<P::Graph>>> {
        self.storage.as_storage()
    }
}

impl<M> Bypass<ModalCore<Immediate<M>>> for FaceMutation<Immediate<M>>
where
    M: Parametric,
{
    fn bypass(self) -> Self::Commit {
        let FaceMutation {
            inner,
            storage: faces,
            ..
        } = self;
        inner.bypass().fuse(faces)
    }
}

// TODO: This is a hack. Replace this with delegation.
impl<P> Deref for FaceMutation<P>
where
    P: Mode,
{
    type Target = EdgeMutation<P>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<P> DerefMut for FaceMutation<P>
where
    P: Mode,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<P> From<ModalCore<P>> for FaceMutation<P>
where
    P: Mode,
{
    fn from(core: ModalCore<P>) -> Self {
        let (vertices, arcs, edges, faces) = core.unfuse();
        FaceMutation {
            storage: faces,
            inner: Core::empty().fuse(vertices).fuse(arcs).fuse(edges).into(),
        }
    }
}

impl<M> Transact<ModalCore<Immediate<M>>> for FaceMutation<Immediate<M>>
where
    M: Parametric,
{
    type Commit = ModalCore<Immediate<M>>;
    type Abort = ();
    type Error = GraphError;

    // TODO: Ensure that faces are in a consistent state.
    fn commit(self) -> Result<Self::Commit, (Self::Abort, Self::Error)> {
        let FaceMutation {
            inner,
            storage: faces,
            ..
        } = self;
        inner.commit().map(move |core| core.fuse(faces))
    }

    fn abort(self) -> Self::Abort {}
}

impl<M> Transact<ModalCore<Transacted<M>>> for FaceMutation<Transacted<M>>
where
    M: Parametric,
{
    type Commit = ModalCore<Transacted<M>>;
    type Abort = ModalCore<Transacted<M>>;
    type Error = GraphError;

    // TODO: Ensure that faces are in a consistent state.
    fn commit(self) -> Result<Self::Commit, (Self::Abort, Self::Error)> {
        let FaceMutation {
            inner,
            storage: faces,
            ..
        } = self;
        match inner.commit() {
            Ok(core) => Ok(core.fuse(faces)),
            Err((core, error)) => Err((core.fuse(faces), error)),
        }
    }

    fn abort(self) -> Self::Abort {
        let FaceMutation {
            inner,
            storage: faces,
            ..
        } = self;
        inner.abort().fuse(faces)
    }
}

pub struct FaceInsertCache {
    perimeter: SmallVec<[VertexKey; 4]>,
    connectivity: (
        HashMap<VertexKey, Vec<ArcKey>>,
        HashMap<VertexKey, Vec<ArcKey>>,
    ),
}

impl FaceInsertCache {
    pub fn from_ring<B, T>(ring: T) -> Result<Self, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Data<B>>>
            + AsStorage<Face<Data<B>>>
            + AsStorage<Vertex<Data<B>>>
            + Consistent
            + Parametric,
        T: ToRing<B>,
    {
        let ring = ring.into_ring();
        let (storage, _) = ring.arc().unbind();
        FaceInsertCache::from_storage(storage, ring.vertices().keys())
    }

    pub fn from_storage<B, K>(storage: B, perimeter: K) -> Result<Self, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Data<B>>>
            + AsStorage<Face<Data<B>>>
            + AsStorage<Vertex<Data<B>>>
            + Parametric,
        K: IntoIterator,
        K::Item: Borrow<VertexKey>,
    {
        let perimeter = perimeter
            .into_iter()
            .map(|key| *key.borrow())
            .collect::<SmallVec<_>>();
        let arity = perimeter.len();
        let set = perimeter.iter().cloned().collect::<HashSet<_>>();
        if set.len() != arity {
            // Vertex keys are not unique.
            return Err(GraphError::TopologyMalformed);
        }

        let storage = storage.reborrow();
        let vertices = perimeter
            .iter()
            .cloned()
            .map(|key| VertexView::bind(storage, key).ok_or_else(|| GraphError::TopologyNotFound))
            .collect::<Result<SmallVec<[_; 4]>, _>>()?;
        for (previous, next) in perimeter
            .iter()
            .cloned()
            .perimeter()
            .map(|keys| ArcView::bind(storage, keys.into()))
            .perimeter()
        {
            if let Some(previous) = previous {
                if previous.face.is_some() {
                    // A face already occupies an interior arc.
                    return Err(GraphError::TopologyConflict);
                }
                // Let the previous arc be AB and the next arc be BC. The
                // vertices A, B, and C lie within the implied ring in order.
                //
                // If BC does not exist and AB is adjacent to some arc BX, then
                // X must not lie within the implied ring (the ordered set of
                // vertices given to this function). If X is within the path,
                // then BX must bisect the implied ring (because X cannot be C).
                if next.is_none() {
                    if let Some(next) = previous.into_reachable_next_arc() {
                        let (_, destination) = next.key().into();
                        if set.contains(&destination) {
                            return Err(GraphError::TopologyConflict);
                        }
                    }
                }
            }
        }

        let mut incoming = HashMap::with_capacity(arity);
        let mut outgoing = HashMap::with_capacity(arity);
        for vertex in vertices {
            let key = vertex.key();
            incoming.insert(key, vertex.reachable_incoming_arcs().keys().collect());
            outgoing.insert(key, vertex.reachable_outgoing_arcs().keys().collect());
        }
        Ok(FaceInsertCache {
            perimeter,
            connectivity: (incoming, outgoing),
        })
    }
}

pub struct FaceRemoveCache {
    abc: FaceKey,
    arcs: Vec<ArcKey>,
}

impl FaceRemoveCache {
    // TODO: Should this require consistency?
    pub fn from_face<B>(face: FaceView<B>) -> Result<Self, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Data<B>>>
            + AsStorage<Face<Data<B>>>
            + AsStorage<Vertex<Data<B>>>
            + Consistent
            + Parametric,
    {
        let arcs = face.adjacent_arcs().keys().collect();
        Ok(FaceRemoveCache {
            abc: face.key(),
            arcs,
        })
    }
}

pub struct FaceSplitCache {
    cache: FaceRemoveCache,
    left: Vec<VertexKey>,
    right: Vec<VertexKey>,
}

impl FaceSplitCache {
    pub fn from_face<B>(
        face: FaceView<B>,
        source: VertexKey,
        destination: VertexKey,
    ) -> Result<Self, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Data<B>>>
            + AsStorage<Face<Data<B>>>
            + AsStorage<Vertex<Data<B>>>
            + Consistent
            + Parametric,
    {
        let perimeter = |face: FaceView<_>| {
            face.adjacent_vertices()
                .keys()
                .collect::<Vec<_>>()
                .into_iter()
                .cycle()
        };
        // This closure determines if any arcs in the given perimeter are
        // connected to a face other than the face initiating the split. This
        // may occur if an adjacent face shares two or more edges with the
        // initiating face and the source and destination vertices of the split
        // lie along that boundary.
        let is_intersecting = |perimeter: &[_]| {
            for (a, b) in perimeter.iter().cloned().perimeter() {
                let ab = (a, b).into();
                if let Some(arc) = Rebind::<_, ArcView<_>>::rebind(face.to_ref(), ab) {
                    if let Some(key) = arc.face().map(|face| face.key()) {
                        if key != face.key() {
                            return true;
                        }
                    }
                }
            }
            false
        };
        face.shortest_metric(source.into(), destination.into())
            .and_then(|metric| {
                if metric <= 1 {
                    Err(GraphError::TopologyMalformed)
                }
                else {
                    Ok(())
                }
            })?;
        // Note that the winding of the perimeters must be relatively oriented.
        let left = perimeter(face.to_ref())
            .perimeter()
            .skip_while(|(_, b)| *b != source)
            .take_while(|(a, _)| *a != destination)
            .map(|(_, b)| b)
            .collect::<Vec<_>>();
        let right = perimeter(face.to_ref())
            .perimeter()
            .skip_while(|(_, b)| *b != destination)
            .take_while(|(a, _)| *a != source)
            .map(|(_, b)| b)
            .collect::<Vec<_>>();
        if is_intersecting(&left) || is_intersecting(&right) {
            return Err(GraphError::TopologyConflict);
        }
        Ok(FaceSplitCache {
            cache: FaceRemoveCache::from_face(face)?,
            left,
            right,
        })
    }
}

pub struct FacePokeCache {
    vertices: Vec<VertexKey>,
    cache: FaceRemoveCache,
}

impl FacePokeCache {
    pub fn from_face<B>(face: FaceView<B>) -> Result<Self, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Data<B>>>
            + AsStorage<Face<Data<B>>>
            + AsStorage<Vertex<Data<B>>>
            + Consistent
            + Parametric,
    {
        let vertices = face.adjacent_vertices().keys().collect();
        Ok(FacePokeCache {
            vertices,
            cache: FaceRemoveCache::from_face(face)?,
        })
    }
}

pub struct FaceBridgeCache {
    source: SmallVec<[ArcKey; 4]>,
    destination: SmallVec<[ArcKey; 4]>,
    cache: (FaceRemoveCache, FaceRemoveCache),
}

impl FaceBridgeCache {
    pub fn from_face<B>(face: FaceView<B>, destination: FaceKey) -> Result<Self, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Data<B>>>
            + AsStorage<Face<Data<B>>>
            + AsStorage<Vertex<Data<B>>>
            + Consistent
            + Parametric,
    {
        let destination: FaceView<_> = face
            .to_ref()
            .rebind(destination)
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        let cache = (
            FaceRemoveCache::from_face(face.to_ref())?,
            FaceRemoveCache::from_face(destination.to_ref())?,
        );
        // Ensure that the opposite face exists and has the same arity.
        if face.arity() != destination.arity() {
            return Err(GraphError::ArityNonUniform);
        }
        Ok(FaceBridgeCache {
            source: face.adjacent_arcs().keys().collect(),
            destination: destination.adjacent_arcs().keys().collect(),
            cache,
        })
    }
}

pub struct FaceExtrudeCache {
    sources: Vec<VertexKey>,
    //destinations: Vec<G::Vertex>,
    //geometry: G::Face,
    cache: FaceRemoveCache,
}

impl FaceExtrudeCache {
    pub fn from_face<B>(face: FaceView<B>) -> Result<Self, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Data<B>>>
            + AsStorage<Face<Data<B>>>
            + AsStorage<Vertex<Data<B>>>
            + Consistent
            + Parametric,
    {
        let sources = face.adjacent_vertices().keys().collect();
        let cache = FaceRemoveCache::from_face(face)?;
        Ok(FaceExtrudeCache { sources, cache })
    }
}

// TODO: Should this accept arc geometry at all?
pub fn insert_with<N, P, F>(
    mut mutation: N,
    cache: FaceInsertCache,
    f: F,
) -> Result<FaceKey, GraphError>
where
    N: AsMut<Mutation<P>>,
    P: Mode,
    P::Graph: Mutable,
    F: FnOnce() -> (
        <Data<P::Graph> as GraphData>::Arc,
        <Data<P::Graph> as GraphData>::Face,
    ),
{
    let FaceInsertCache {
        perimeter,
        connectivity,
    } = cache;
    let geometry = f();
    // Insert edges and collect the interior arcs.
    let arcs = perimeter
        .iter()
        .cloned()
        .perimeter()
        .map(|(a, b)| {
            edge::get_or_insert_with(mutation.as_mut(), (a, b), || {
                (Default::default(), geometry.0)
            })
            .map(|(_, (ab, _))| ab)
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Insert the face.
    let face = mutation
        .as_mut()
        .storage
        .as_storage_mut()
        .insert(Face::new(arcs[0], geometry.1));
    mutation.as_mut().connect_face_interior(&arcs, face)?;
    mutation
        .as_mut()
        .connect_face_exterior(&arcs, connectivity)?;
    Ok(face)
}

// TODO: Does this require a cache (or consistency)?
// TODO: This may need to be more destructive to maintain consistency. Edges,
//       arcs, and vertices may also need to be removed.
pub fn remove<N, P>(
    mut mutation: N,
    cache: FaceRemoveCache,
) -> Result<Face<Data<P::Graph>>, GraphError>
where
    N: AsMut<Mutation<P>>,
    P: Mode,
    P::Graph: Mutable,
{
    let FaceRemoveCache { abc, arcs } = cache;
    mutation.as_mut().disconnect_face_interior(&arcs)?;
    let face = mutation
        .as_mut()
        .storage
        .as_storage_mut()
        .remove(&abc)
        .ok_or_else(|| GraphError::TopologyNotFound)?;
    Ok(face)
}

pub fn split<N, P>(mut mutation: N, cache: FaceSplitCache) -> Result<ArcKey, GraphError>
where
    N: AsMut<Mutation<P>>,
    P: Mode,
    P::Graph: Mutable,
{
    let FaceSplitCache { cache, left, right } = cache;
    remove(mutation.as_mut(), cache)?;
    let ab = (left[0], right[0]).into();
    let left = FaceInsertCache::from_storage(mutation.as_mut(), left)?;
    let right = FaceInsertCache::from_storage(mutation.as_mut(), right)?;
    insert_with(mutation.as_mut(), left, Default::default)?;
    insert_with(mutation.as_mut(), right, Default::default)?;
    Ok(ab)
}

pub fn poke_with<N, P, F>(
    mut mutation: N,
    cache: FacePokeCache,
    f: F,
) -> Result<VertexKey, GraphError>
where
    N: AsMut<Mutation<P>>,
    P: Mode,
    P::Graph: Mutable,
    F: FnOnce() -> <Data<P::Graph> as GraphData>::Vertex,
{
    let FacePokeCache { vertices, cache } = cache;
    let face = remove(mutation.as_mut(), cache)?;
    let c = vertex::insert(mutation.as_mut(), f());
    for (a, b) in vertices.into_iter().perimeter() {
        let cache = FaceInsertCache::from_storage(mutation.as_mut(), &[a, b, c])?;
        insert_with(mutation.as_mut(), cache, || (Default::default(), face.data))?;
    }
    Ok(c)
}

pub fn bridge<N, P>(mut mutation: N, cache: FaceBridgeCache) -> Result<(), GraphError>
where
    N: AsMut<Mutation<P>>,
    P: Mode,
    P::Graph: Mutable,
{
    let FaceBridgeCache {
        source,
        destination,
        cache,
    } = cache;
    // Remove the source and destination faces. Pair the topology with edge
    // geometry for the source face.
    remove(mutation.as_mut(), cache.0)?;
    remove(mutation.as_mut(), cache.1)?;
    // TODO: Is it always correct to reverse the order of the opposite face's
    //       arcs?
    // Re-insert the arcs of the faces and bridge the mutual arcs.
    for (ab, cd) in source.into_iter().zip(destination.into_iter().rev()) {
        let cache = ArcBridgeCache::from_storage(mutation.as_mut(), ab, cd)?;
        edge::bridge(mutation.as_mut(), cache)?;
    }
    // TODO: Is there any reasonable entity this can return?
    Ok(())
}

pub fn extrude_with<N, P, F>(
    mut mutation: N,
    cache: FaceExtrudeCache,
    f: F,
) -> Result<FaceKey, GraphError>
where
    N: AsMut<Mutation<P>>,
    P: Mode,
    P::Graph: Mutable,
    F: Fn(<Data<P::Graph> as GraphData>::Vertex) -> <Data<P::Graph> as GraphData>::Vertex,
{
    let FaceExtrudeCache { sources, cache } = cache;
    remove(mutation.as_mut(), cache)?;
    let destinations = {
        let mutation = &*mutation.as_mut();
        sources
            .iter()
            .cloned()
            .flat_map(|a| VertexView::bind(mutation, a))
            .map(|source| f(source.data))
            .collect::<Vec<_>>()
    };
    if sources.len() != destinations.len() {
        return Err(GraphError::TopologyNotFound);
    }
    let destinations = destinations
        .into_iter()
        .map(|geometry| vertex::insert(mutation.as_mut(), geometry))
        .collect::<Vec<_>>();
    // Use the keys for the existing vertices and the translated geometries to
    // construct the extruded face and its connective faces.
    let cache = FaceInsertCache::from_storage(mutation.as_mut(), &destinations)?;
    let extrusion = insert_with(mutation.as_mut(), cache, Default::default)?;
    for ((a, c), (b, d)) in sources
        .into_iter()
        .zip(destinations.into_iter())
        .perimeter()
    {
        let cache = FaceInsertCache::from_storage(mutation.as_mut(), &[a, b, d, c])?;
        // TODO: Split these faces to form triangles.
        insert_with(mutation.as_mut(), cache, Default::default)?;
    }
    Ok(extrusion)
}
