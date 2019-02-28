use std::ops::{Add, Deref, DerefMut, Mul};

use crate::geometry::alias::{ScaledArcNormal, VertexPosition};
use crate::geometry::convert::AsPosition;
use crate::geometry::Geometry;
use crate::graph::container::{Bind, Consistent, Core, Reborrow};
use crate::graph::geometry::ArcNormal;
use crate::graph::mutation::alias::Mutable;
use crate::graph::mutation::face::{self, FaceRemoveCache};
use crate::graph::mutation::vertex::VertexMutation;
use crate::graph::mutation::{Mutate, Mutation};
use crate::graph::payload::{ArcPayload, EdgePayload, FacePayload, VertexPayload};
use crate::graph::storage::convert::alias::*;
use crate::graph::storage::convert::AsStorage;
use crate::graph::storage::{ArcKey, EdgeKey, FaceKey, Storage, VertexKey};
use crate::graph::view::convert::FromKeyedSource;
use crate::graph::view::ArcView;
use crate::graph::GraphError;
use crate::IteratorExt;

pub struct EdgeMutation<G>
where
    G: Geometry,
{
    mutation: VertexMutation<G>,
    storage: (Storage<ArcPayload<G>>, Storage<EdgePayload<G>>),
}

impl<G> EdgeMutation<G>
where
    G: Geometry,
{
    pub fn get_or_insert_edge(
        &mut self,
        span: (VertexKey, VertexKey),
    ) -> Result<(EdgeKey, (ArcKey, ArcKey)), GraphError> {
        self.get_or_insert_edge_with(span, || Default::default())
    }

    pub fn get_or_insert_edge_with<F>(
        &mut self,
        span: (VertexKey, VertexKey),
        f: F,
    ) -> Result<(EdgeKey, (ArcKey, ArcKey)), GraphError>
    where
        F: Clone + FnOnce() -> G::Arc,
    {
        fn get_or_insert_arc_with<G, F>(
            mutation: &mut EdgeMutation<G>,
            span: (VertexKey, VertexKey),
            f: F,
        ) -> (Option<EdgeKey>, ArcKey)
        where
            G: Geometry,
            F: FnOnce() -> G::Arc,
        {
            let (a, _) = span;
            let ab = span.into();
            if let Some(arc) = mutation.storage.0.get(&ab) {
                (arc.edge, ab)
            }
            else {
                mutation
                    .storage
                    .0
                    .insert_with_key(&ab, ArcPayload::new(f()));
                let _ = mutation.connect_outgoing_arc(a, ab);
                (None, ab)
            }
        }

        let (a, b) = span;
        let (e1, ab) = get_or_insert_arc_with(self, (a, b), f.clone());
        let (e2, ba) = get_or_insert_arc_with(self, (b, a), f);
        match (e1, e2) {
            (Some(e1), Some(e2)) if e1 == e2 => Ok((e1, (ab, ba))),
            (None, None) => {
                let ab_ba = self
                    .storage
                    .1
                    .insert(EdgePayload::new(ab, Default::default()));
                self.connect_arc_to_edge(ab, ab_ba)?;
                self.connect_arc_to_edge(ba, ab_ba)?;
                Ok((ab_ba, (ab, ba)))
            }
            // It should not be possible to insert or remove individual arcs
            // and mutations should not allow arcs to be assigned to edges
            // independently of their opposite arcs.
            _ => Err(GraphError::TopologyMalformed),
        }
    }

    pub fn connect_neighboring_arcs(&mut self, ab: ArcKey, bc: ArcKey) -> Result<(), GraphError> {
        self.with_arc_mut(ab, |arc| arc.next = Some(bc))?;
        self.with_arc_mut(bc, |arc| arc.previous = Some(ab))?;
        Ok(())
    }

    pub fn disconnect_next_arc(&mut self, ab: ArcKey) -> Result<Option<ArcKey>, GraphError> {
        let bx = self.with_arc_mut(ab, |arc| arc.next.take())?;
        if let Some(bx) = bx.as_ref() {
            self.with_arc_mut(*bx, |arc| arc.previous.take())
                .map_err(|_| GraphError::TopologyMalformed)?;
        }
        Ok(bx)
    }

    pub fn disconnect_previous_arc(&mut self, ab: ArcKey) -> Result<Option<ArcKey>, GraphError> {
        let xa = self.with_arc_mut(ab, |arc| arc.previous.take())?;
        if let Some(xa) = xa.as_ref() {
            self.with_arc_mut(*xa, |arc| arc.next.take())
                .map_err(|_| GraphError::TopologyMalformed)?;
        }
        Ok(xa)
    }

    pub fn connect_arc_to_edge(&mut self, ab: ArcKey, ab_ba: EdgeKey) -> Result<(), GraphError> {
        self.with_arc_mut(ab, |arc| arc.edge = Some(ab_ba))
    }

    pub fn connect_arc_to_face(&mut self, ab: ArcKey, abc: FaceKey) -> Result<(), GraphError> {
        self.with_arc_mut(ab, |arc| arc.face = Some(abc))
    }

    pub fn disconnect_arc_from_face(&mut self, ab: ArcKey) -> Result<Option<FaceKey>, GraphError> {
        self.with_arc_mut(ab, |arc| arc.face.take())
    }

    fn with_arc_mut<T, F>(&mut self, ab: ArcKey, mut f: F) -> Result<T, GraphError>
    where
        F: FnMut(&mut ArcPayload<G>) -> T,
    {
        let arc = self
            .storage
            .0
            .get_mut(&ab)
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        Ok(f(arc))
    }
}

impl<G> AsStorage<ArcPayload<G>> for EdgeMutation<G>
where
    G: Geometry,
{
    fn as_storage(&self) -> &Storage<ArcPayload<G>> {
        &self.storage.0
    }
}

impl<G> AsStorage<EdgePayload<G>> for EdgeMutation<G>
where
    G: Geometry,
{
    fn as_storage(&self) -> &Storage<EdgePayload<G>> {
        &self.storage.1
    }
}

impl<G> Mutate for EdgeMutation<G>
where
    G: Geometry,
{
    type Mutant =
        Core<Storage<VertexPayload<G>>, Storage<ArcPayload<G>>, Storage<EdgePayload<G>>, ()>;
    type Error = GraphError;

    fn mutate(mutant: Self::Mutant) -> Self {
        let (vertices, arcs, edges, ..) = mutant.into_storage();
        EdgeMutation {
            mutation: VertexMutation::mutate(Core::empty().bind(vertices)),
            storage: (arcs, edges),
        }
    }

    fn commit(self) -> Result<Self::Mutant, Self::Error> {
        let EdgeMutation {
            mutation,
            storage: (arcs, edges),
            ..
        } = self;
        mutation.commit().and_then(move |core| {
            let (vertices, ..) = core.into_storage();
            Ok(Core::empty().bind(vertices).bind(arcs).bind(edges))
        })
    }
}

impl<G> Deref for EdgeMutation<G>
where
    G: Geometry,
{
    type Target = VertexMutation<G>;

    fn deref(&self) -> &Self::Target {
        &self.mutation
    }
}

impl<G> DerefMut for EdgeMutation<G>
where
    G: Geometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.mutation
    }
}

struct ArcRemoveCache<G>
where
    G: Geometry,
{
    ab: ArcKey,
    xa: Option<ArcKey>,
    bx: Option<ArcKey>,
    cache: Option<FaceRemoveCache<G>>,
}

impl<G> ArcRemoveCache<G>
where
    G: Geometry,
{
    pub fn snapshot<M>(storage: M, ab: ArcKey) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<ArcPayload<G>>
            + AsStorage<FacePayload<G>>
            + AsStorage<VertexPayload<G>>
            + Consistent,
    {
        let storage = storage.reborrow();
        let arc = ArcView::from_keyed_source((ab, storage))
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        // If the edge has no neighbors, then `xa` and `bx` will refer to the
        // opposite arc of `ab`. In this case, the vertices `a` and `b` should
        // have no leading arcs after the removal. The cache will have its `xa`
        // and `bx` fields set to `None` in this case.
        let ba = arc.opposite_arc().key();
        let xa = arc.previous_arc().key();
        let bx = arc.next_arc().key();
        let cache = if let Some(face) = arc.face() {
            Some(FaceRemoveCache::snapshot(storage, face.key())?)
        }
        else {
            None
        };
        Ok(ArcRemoveCache {
            ab,
            xa: if xa != ba { Some(xa) } else { None },
            bx: if bx != ba { Some(bx) } else { None },
            cache,
        })
    }
}

pub struct EdgeRemoveCache<G>
where
    G: Geometry,
{
    a: VertexKey,
    b: VertexKey,
    ab_ba: EdgeKey,
    arc: ArcRemoveCache<G>,
    opposite: ArcRemoveCache<G>,
}

impl<G> EdgeRemoveCache<G>
where
    G: Geometry,
{
    pub fn snapshot<M>(storage: M, ab: ArcKey) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<ArcPayload<G>>
            + AsStorage<EdgePayload<G>>
            + AsStorage<FacePayload<G>>
            + AsStorage<VertexPayload<G>>
            + Consistent,
    {
        let storage = storage.reborrow();
        let arc = ArcView::from_keyed_source((ab, storage))
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        let a = arc.source_vertex().key();
        let b = arc.destination_vertex().key();
        let ba = arc.opposite_arc().key();
        let ab_ba = arc.edge().key();
        Ok(EdgeRemoveCache {
            a,
            b,
            ab_ba,
            arc: ArcRemoveCache::snapshot(storage, ab)?,
            opposite: ArcRemoveCache::snapshot(storage, ba)?,
        })
    }
}

pub struct EdgeSplitCache<G>
where
    G: Geometry,
{
    a: VertexKey,
    b: VertexKey,
    ab: ArcKey,
    ba: ArcKey,
    ab_ba: EdgeKey,
    geometry: G::Vertex,
}

impl<G> EdgeSplitCache<G>
where
    G: Geometry,
{
    pub fn snapshot<M>(storage: M, ab: ArcKey, geometry: G::Vertex) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target:
            AsStorage<ArcPayload<G>> + AsStorage<EdgePayload<G>> + AsStorage<VertexPayload<G>>,
    {
        let storage = storage.reborrow();
        let arc = ArcView::from_keyed_source((ab, storage))
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        let opposite = arc
            .reachable_opposite_arc()
            .ok_or_else(|| GraphError::TopologyMalformed)?;
        let source = opposite
            .reachable_destination_vertex()
            .ok_or_else(|| GraphError::TopologyMalformed)?;
        let destination = arc
            .reachable_destination_vertex()
            .ok_or_else(|| GraphError::TopologyMalformed)?;
        let edge = arc
            .reachable_edge()
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        Ok(EdgeSplitCache {
            a: source.key(),
            b: destination.key(),
            ab: arc.key(),
            ba: opposite.key(),
            ab_ba: edge.key(),
            geometry,
        })
    }
}

pub struct ArcBridgeCache<G>
where
    G: Geometry,
{
    a: VertexKey,
    b: VertexKey,
    c: VertexKey,
    d: VertexKey,
    arc: G::Arc,
    face: G::Face,
}

impl<G> ArcBridgeCache<G>
where
    G: Geometry,
{
    pub fn snapshot<M>(storage: M, source: ArcKey, destination: ArcKey) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target:
            AsStorage<ArcPayload<G>> + AsStorage<FacePayload<G>> + AsStorage<VertexPayload<G>>,
    {
        let storage = storage.reborrow();
        let source = ArcView::from_keyed_source((source, storage))
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        let destination = ArcView::from_keyed_source((destination, storage))
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        let a = source
            .reachable_source_vertex()
            .ok_or_else(|| GraphError::TopologyMalformed)?
            .key();
        let b = source
            .reachable_destination_vertex()
            .ok_or_else(|| GraphError::TopologyMalformed)?
            .key();
        let c = destination
            .reachable_source_vertex()
            .ok_or_else(|| GraphError::TopologyMalformed)?
            .key();
        let d = destination
            .reachable_destination_vertex()
            .ok_or_else(|| GraphError::TopologyMalformed)?
            .key();
        // At this point, we can assume the vertices a, b, c, and d exist in
        // the mesh. Before mutating the mesh, ensure that existing interior
        // arcs are boundaries.
        for arc in [a, b, c, d]
            .into_iter()
            .cloned()
            .perimeter()
            .flat_map(|ab| ArcView::from_keyed_source((ab.into(), storage)))
        {
            if !arc.is_boundary_arc() {
                return Err(GraphError::TopologyConflict);
            }
        }
        Ok(ArcBridgeCache {
            a,
            b,
            c,
            d,
            arc: source.geometry.clone(),
            face: source
                .reachable_opposite_arc()
                .and_then(|opposite| opposite.into_reachable_face())
                .map(|face| face.geometry.clone())
                .unwrap_or_else(Default::default),
        })
    }
}

pub struct ArcExtrudeCache<G>
where
    G: Geometry,
{
    ab: ArcKey,
    vertices: (G::Vertex, G::Vertex),
    arc: G::Arc,
}

impl<G> ArcExtrudeCache<G>
where
    G: Geometry,
{
    pub fn snapshot<M, T>(storage: M, ab: ArcKey, distance: T) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<ArcPayload<G>>
            + AsStorage<FacePayload<G>>
            + AsStorage<VertexPayload<G>>
            + Consistent,
        G: Geometry + ArcNormal,
        G::Normal: Mul<T>,
        G::Vertex: AsPosition,
        ScaledArcNormal<G, T>: Clone,
        VertexPosition<G>: Add<ScaledArcNormal<G, T>, Output = VertexPosition<G>> + Clone,
    {
        // Get the extruded geometry.
        let (vertices, arc) = {
            let arc = ArcView::from_keyed_source((ab, storage))
                .ok_or_else(|| GraphError::TopologyNotFound)?;
            if !arc.is_boundary_arc() {
                return Err(GraphError::TopologyConflict.into());
            }
            let mut vertices = (
                arc.reachable_destination_vertex()
                    .ok_or_else(|| GraphError::TopologyConflict)?
                    .geometry
                    .clone(),
                arc.reachable_source_vertex()
                    .ok_or_else(|| GraphError::TopologyConflict)?
                    .geometry
                    .clone(),
            );
            let translation = arc.normal()? * distance;
            *vertices.0.as_position_mut() = vertices.0.as_position().clone() + translation.clone();
            *vertices.1.as_position_mut() = vertices.1.as_position().clone() + translation;
            (vertices, arc.geometry.clone())
        };
        Ok(ArcExtrudeCache { ab, vertices, arc })
    }
}

// TODO: Removing arcs must also remove disjoint vertices. More importantly,
//       the leading arc of vertices may be invalidated by this operation and
//       must be healed. This code does not handle these cases, and so can
//       become inconsistent.
pub fn remove_with_cache<M, N, G>(
    mut mutation: N,
    cache: EdgeRemoveCache<G>,
) -> Result<(EdgePayload<G>, (ArcPayload<G>, ArcPayload<G>)), GraphError>
where
    N: AsMut<Mutation<M, G>>,
    M: Mutable<G>,
    G: Geometry,
{
    fn remove_arc_with_cache<M, N, G>(
        mut mutation: N,
        cache: ArcRemoveCache<G>,
    ) -> Result<ArcPayload<G>, GraphError>
    where
        N: AsMut<Mutation<M, G>>,
        M: Mutable<G>,
        G: Geometry,
    {
        let ArcRemoveCache { ab, cache, .. } = cache;
        if let Some(cache) = cache {
            face::remove_with_cache(mutation.as_mut(), cache)?;
        }
        mutation
            .as_mut()
            .storage
            .0
            .remove(&ab)
            .ok_or_else(|| GraphError::TopologyNotFound)
    }

    let EdgeRemoveCache {
        a,
        b,
        ab_ba,
        arc,
        opposite,
        ..
    } = cache;
    // Connect each vertex to a remaining outgoing edge.
    if let Some(ax) = opposite.bx {
        mutation.as_mut().connect_outgoing_arc(a, ax)?;
    }
    if let Some(bx) = arc.bx {
        mutation.as_mut().connect_outgoing_arc(b, bx)?;
    }
    // Connect previous and next arcs across the edge to be removed.
    if let (Some(xa), Some(ax)) = (arc.xa, opposite.bx) {
        mutation.as_mut().connect_neighboring_arcs(xa, ax)?;
    }
    if let (Some(xb), Some(bx)) = (opposite.xa, arc.bx) {
        mutation.as_mut().connect_neighboring_arcs(xb, bx)?;
    }
    let edge = mutation
        .as_mut()
        .storage
        .1
        .remove(&ab_ba)
        .ok_or_else(|| GraphError::TopologyNotFound)?;
    Ok((
        edge,
        (
            remove_arc_with_cache(mutation.as_mut(), arc)?,
            remove_arc_with_cache(mutation.as_mut(), opposite)?,
        ),
    ))
}

pub fn split_with_cache<M, N, G>(
    mut mutation: N,
    cache: EdgeSplitCache<G>,
) -> Result<VertexKey, GraphError>
where
    N: AsMut<Mutation<M, G>>,
    M: Mutable<G>,
    G: Geometry,
{
    fn remove<M, N, G>(mut mutation: N, ab: ArcKey) -> Result<ArcPayload<G>, GraphError>
    where
        N: AsMut<Mutation<M, G>>,
        M: Mutable<G>,
        G: Geometry,
    {
        let (a, _) = ab.into();
        mutation.as_mut().disconnect_outgoing_arc(a)?;
        let xa = mutation.as_mut().disconnect_previous_arc(ab)?;
        let bx = mutation.as_mut().disconnect_next_arc(ab)?;
        let mut arc = mutation.as_mut().storage.0.remove(&ab).unwrap();
        // Restore the connectivity of the arc. The mutations will clear this
        // data, because it is still a part of the mesh at that point.
        arc.previous = xa;
        arc.next = bx;
        Ok(arc)
    }

    fn split_at_vertex<M, N, G>(
        mut mutation: N,
        a: VertexKey,
        b: VertexKey,
        m: VertexKey,
        ab: ArcKey,
    ) -> Result<(ArcKey, ArcKey), GraphError>
    where
        N: AsMut<Mutation<M, G>>,
        M: Mutable<G>,
        G: Geometry,
    {
        // Remove the arc and insert two truncated arcs in its place.
        let ArcPayload {
            next,
            previous,
            face,
            geometry,
            ..
        } = remove(mutation.as_mut(), ab)?;
        let am = mutation
            .as_mut()
            .get_or_insert_edge_with((a, m), || geometry.clone())
            .map(|(_, (am, _))| am)?;
        let mb = mutation
            .as_mut()
            .get_or_insert_edge_with((m, b), move || geometry)
            .map(|(_, (mb, _))| mb)?;
        // Connect the new arcs to each other and their leading arcs.
        mutation.as_mut().connect_neighboring_arcs(am, mb)?;
        if let Some(xa) = previous {
            mutation.as_mut().connect_neighboring_arcs(xa, am)?;
        }
        if let Some(bx) = next {
            mutation.as_mut().connect_neighboring_arcs(mb, bx)?;
        }
        // Update the associated face, if any, because it may refer to the
        // removed arc.
        if let Some(abc) = face {
            mutation.as_mut().connect_face_to_arc(am, abc)?;
            mutation.as_mut().connect_arc_to_face(am, abc)?;
            mutation.as_mut().connect_arc_to_face(mb, abc)?;
        }
        Ok((am, mb))
    }

    let EdgeSplitCache {
        a,
        b,
        ab,
        ba,
        ab_ba,
        geometry,
        ..
    } = cache;
    let m = mutation.as_mut().insert_vertex(geometry);
    // Remove the edge.
    let _ = mutation
        .as_mut()
        .storage
        .1
        .remove(&ab_ba)
        .ok_or_else(|| GraphError::TopologyMalformed)?;
    // Split the arcs.
    split_at_vertex(mutation.as_mut(), a, b, m, ab)?;
    split_at_vertex(mutation.as_mut(), b, a, m, ba)?;
    Ok(m)
}

pub fn bridge_with_cache<M, N, G>(
    mut mutation: N,
    cache: ArcBridgeCache<G>,
) -> Result<FaceKey, GraphError>
where
    N: AsMut<Mutation<M, G>>,
    M: Mutable<G>,
    G: Geometry,
{
    let ArcBridgeCache {
        a,
        b,
        c,
        d,
        arc,
        face,
        ..
    } = cache;
    mutation.as_mut().insert_face(&[a, b, c, d], (arc, face))
}

pub fn extrude_with_cache<M, N, G, T>(
    mut mutation: N,
    cache: ArcExtrudeCache<G>,
) -> Result<ArcKey, GraphError>
where
    N: AsMut<Mutation<M, G>>,
    M: Mutable<G>,
    G: Geometry + ArcNormal,
    G::Normal: Mul<T>,
    G::Vertex: AsPosition,
    ScaledArcNormal<G, T>: Clone,
    VertexPosition<G>: Add<ScaledArcNormal<G, T>, Output = VertexPosition<G>> + Clone,
{
    let ArcExtrudeCache {
        ab, vertices, arc, ..
    } = cache;
    let mutation = mutation.as_mut();
    let c = mutation.insert_vertex(vertices.0);
    let d = mutation.insert_vertex(vertices.1);
    // TODO: If this arc already exists, then this should probably return an
    //       error.
    let cd = mutation
        .get_or_insert_edge_with((c, d), move || arc)
        .map(|(_, (cd, _))| cd)?;
    let cache = ArcBridgeCache::snapshot(
        &Core::empty()
            .bind(mutation.as_vertex_storage())
            .bind(mutation.as_arc_storage())
            .bind(mutation.as_face_storage()),
        ab,
        cd,
    )?;
    bridge_with_cache(mutation, cache).map(|_| cd)
}
