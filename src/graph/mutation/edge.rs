use std::ops::{Add, Deref, DerefMut, Mul};

use crate::geometry::alias::{ScaledEdgeLateral, VertexPosition};
use crate::geometry::convert::AsPosition;
use crate::geometry::Geometry;
use crate::graph::container::alias::OwnedCore;
use crate::graph::container::{Bind, Consistent, Core, Reborrow};
use crate::graph::geometry::{EdgeLateral, EdgeMidpoint};
use crate::graph::mutation::face::{self, FaceRemoveCache};
use crate::graph::mutation::vertex::VertexMutation;
use crate::graph::mutation::{Mutate, Mutation};
use crate::graph::storage::convert::alias::*;
use crate::graph::storage::convert::AsStorage;
use crate::graph::storage::{FaceKey, HalfKey, Storage, VertexKey};
use crate::graph::topology::{Face, Half, Vertex};
use crate::graph::view::convert::FromKeyedSource;
use crate::graph::view::HalfView;
use crate::graph::GraphError;
use crate::IteratorExt;

pub struct EdgeMutation<G>
where
    G: Geometry,
{
    mutation: VertexMutation<G>,
    storage: Storage<Half<G>>,
}

impl<G> EdgeMutation<G>
where
    G: Geometry,
{
    pub fn get_or_insert_edge(
        &mut self,
        span: (VertexKey, VertexKey),
    ) -> Result<(HalfKey, HalfKey), GraphError> {
        self.get_or_insert_edge_with(span, || Default::default())
    }

    pub fn get_or_insert_edge_with<F>(
        &mut self,
        span: (VertexKey, VertexKey),
        f: F,
    ) -> Result<(HalfKey, HalfKey), GraphError>
    where
        F: Clone + FnOnce() -> G::Half,
    {
        fn get_or_insert_half_with<G, F>(
            mutation: &mut EdgeMutation<G>,
            span: (VertexKey, VertexKey),
            f: F,
        ) -> HalfKey
        where
            G: Geometry,
            F: FnOnce() -> G::Half,
        {
            let (a, _) = span;
            let ab = span.into();
            if mutation.storage.contains_key(&ab) {
                ab
            }
            else {
                mutation.storage.insert_with_key(&ab, Half::new(f()));
                let _ = mutation.connect_outgoing_half(a, ab);
                ab
            }
        }

        let (a, b) = span;
        Ok((
            get_or_insert_half_with(self, (a, b), f.clone()),
            get_or_insert_half_with(self, (b, a), f),
        ))
    }

    pub fn connect_neighboring_halves(
        &mut self,
        ab: HalfKey,
        bc: HalfKey,
    ) -> Result<(), GraphError> {
        self.storage
            .get_mut(&ab)
            .ok_or_else(|| GraphError::TopologyNotFound)?
            .next = Some(bc);
        self.storage
            .get_mut(&bc)
            .ok_or_else(|| GraphError::TopologyNotFound)?
            .previous = Some(ab);
        Ok(())
    }

    pub fn disconnect_next_half(&mut self, ab: HalfKey) -> Result<Option<HalfKey>, GraphError> {
        let bx = {
            self.storage
                .get_mut(&ab)
                .ok_or_else(|| GraphError::TopologyNotFound)?
                .next
                .take()
        };
        if let Some(bx) = bx.as_ref() {
            self.storage
                .get_mut(bx)
                .ok_or_else(|| GraphError::TopologyMalformed)?
                .previous
                .take();
        }
        Ok(bx)
    }

    pub fn disconnect_previous_half(&mut self, ab: HalfKey) -> Result<Option<HalfKey>, GraphError> {
        let xa = {
            self.storage
                .get_mut(&ab)
                .ok_or_else(|| GraphError::TopologyNotFound)?
                .previous
                .take()
        };
        if let Some(xa) = xa.as_ref() {
            self.storage
                .get_mut(xa)
                .ok_or_else(|| GraphError::TopologyMalformed)?
                .next
                .take();
        }
        Ok(xa)
    }

    pub fn connect_half_to_face(&mut self, ab: HalfKey, abc: FaceKey) -> Result<(), GraphError> {
        self.storage
            .get_mut(&ab)
            .ok_or_else(|| GraphError::TopologyNotFound)?
            .face = Some(abc);
        Ok(())
    }

    pub fn disconnect_half_from_face(
        &mut self,
        ab: HalfKey,
    ) -> Result<Option<FaceKey>, GraphError> {
        let face = self
            .storage
            .get_mut(&ab)
            .ok_or_else(|| GraphError::TopologyNotFound)?
            .face
            .take();
        Ok(face)
    }
}

impl<G> AsStorage<Half<G>> for EdgeMutation<G>
where
    G: Geometry,
{
    fn as_storage(&self) -> &Storage<Half<G>> {
        &self.storage
    }
}

impl<G> Mutate for EdgeMutation<G>
where
    G: Geometry,
{
    type Mutant = Core<Storage<Vertex<G>>, Storage<Half<G>>, ()>;
    type Error = GraphError;

    fn mutate(mutant: Self::Mutant) -> Self {
        let (vertices, halves, ..) = mutant.into_storage();
        EdgeMutation {
            mutation: VertexMutation::mutate(Core::empty().bind(vertices)),
            storage: halves,
        }
    }

    fn commit(self) -> Result<Self::Mutant, Self::Error> {
        let EdgeMutation {
            mutation,
            storage: halves,
            ..
        } = self;
        mutation.commit().and_then(move |core| {
            let (vertices, ..) = core.into_storage();
            Ok(Core::empty().bind(vertices).bind(halves))
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

struct HalfRemoveCache<G>
where
    G: Geometry,
{
    ab: HalfKey,
    xa: Option<HalfKey>,
    bx: Option<HalfKey>,
    cache: Option<FaceRemoveCache<G>>,
}

impl<G> HalfRemoveCache<G>
where
    G: Geometry,
{
    pub fn snapshot<M>(storage: M, ab: HalfKey) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<Half<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Consistent,
    {
        let storage = storage.reborrow();
        let half = HalfView::from_keyed_source((ab, storage))
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        // If the composite-edge has no neighbors, then `xa` and `bx` will
        // refer to the opposite half-edge of `ab`. In this case, the vertices
        // `a` and `b` should have no leading half-edges after the removal. The
        // cache will have its `xa` and `bx` fields set to `None` in this case.
        let ba = half.opposite_half().key();
        let xa = half.previous_half().key();
        let bx = half.next_half().key();
        let cache = if let Some(face) = half.face() {
            Some(FaceRemoveCache::snapshot(storage, face.key())?)
        }
        else {
            None
        };
        Ok(HalfRemoveCache {
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
    half: HalfRemoveCache<G>,
    opposite: HalfRemoveCache<G>,
}

impl<G> EdgeRemoveCache<G>
where
    G: Geometry,
{
    pub fn snapshot<M>(storage: M, ab: HalfKey) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<Half<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Consistent,
    {
        let storage = storage.reborrow();
        let half = HalfView::from_keyed_source((ab, storage))
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        let a = half.source_vertex().key();
        let b = half.destination_vertex().key();
        let ba = half.opposite_half().key();
        Ok(EdgeRemoveCache {
            a,
            b,
            half: HalfRemoveCache::snapshot(storage, ab)?,
            opposite: HalfRemoveCache::snapshot(storage, ba)?,
        })
    }
}

pub struct EdgeSplitCache<G>
where
    G: Geometry,
{
    a: VertexKey,
    b: VertexKey,
    ab: HalfKey,
    ba: HalfKey,
    midpoint: G::Vertex,
}

impl<G> EdgeSplitCache<G>
where
    G: EdgeMidpoint<Midpoint = VertexPosition<G>> + Geometry,
    G::Vertex: AsPosition,
{
    pub fn snapshot<M>(storage: M, ab: HalfKey) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<Half<G>> + AsStorage<Vertex<G>>,
    {
        let storage = storage.reborrow();
        let half = HalfView::from_keyed_source((ab, storage))
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        let opposite = half
            .reachable_opposite_half()
            .ok_or_else(|| GraphError::TopologyMalformed)?;
        let source = opposite
            .reachable_destination_vertex()
            .ok_or_else(|| GraphError::TopologyMalformed)?;
        let destination = half
            .reachable_destination_vertex()
            .ok_or_else(|| GraphError::TopologyMalformed)?;
        let mut midpoint = source.geometry.clone();
        *midpoint.as_position_mut() = EdgeMidpoint::midpoint(half)?;
        Ok(EdgeSplitCache {
            a: source.key(),
            b: destination.key(),
            ab,
            ba: opposite.key(),
            midpoint,
        })
    }
}

pub struct HalfBridgeCache<G>
where
    G: Geometry,
{
    a: VertexKey,
    b: VertexKey,
    c: VertexKey,
    d: VertexKey,
    half: G::Half,
    face: G::Face,
}

impl<G> HalfBridgeCache<G>
where
    G: Geometry,
{
    pub fn snapshot<M>(
        storage: M,
        source: HalfKey,
        destination: HalfKey,
    ) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<Half<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>>,
    {
        let storage = storage.reborrow();
        let source = HalfView::from_keyed_source((source, storage))
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        let destination = HalfView::from_keyed_source((destination, storage))
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
        // half-edges are boundaries.
        for half in [a, b, c, d]
            .into_iter()
            .cloned()
            .perimeter()
            .flat_map(|ab| HalfView::from_keyed_source((ab.into(), storage)))
        {
            if !half.is_boundary_half() {
                return Err(GraphError::TopologyConflict);
            }
        }
        Ok(HalfBridgeCache {
            a,
            b,
            c,
            d,
            half: source.geometry.clone(),
            face: source
                .reachable_opposite_half()
                .and_then(|opposite| opposite.into_reachable_face())
                .map(|face| face.geometry.clone())
                .unwrap_or_else(Default::default),
        })
    }
}

pub struct HalfExtrudeCache<G>
where
    G: Geometry,
{
    ab: HalfKey,
    vertices: (G::Vertex, G::Vertex),
    half: G::Half,
}

impl<G> HalfExtrudeCache<G>
where
    G: Geometry,
{
    pub fn snapshot<M, T>(storage: M, ab: HalfKey, distance: T) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<Half<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Consistent,
        G: Geometry + EdgeLateral,
        G::Lateral: Mul<T>,
        G::Vertex: AsPosition,
        ScaledEdgeLateral<G, T>: Clone,
        VertexPosition<G>: Add<ScaledEdgeLateral<G, T>, Output = VertexPosition<G>> + Clone,
    {
        // Get the extruded geometry.
        let (vertices, half) = {
            let half = HalfView::from_keyed_source((ab, storage))
                .ok_or_else(|| GraphError::TopologyNotFound)?;
            if !half.is_boundary_half() {
                return Err(GraphError::TopologyConflict.into());
            }
            let mut vertices = (
                half.reachable_destination_vertex()
                    .ok_or_else(|| GraphError::TopologyConflict)?
                    .geometry
                    .clone(),
                half.reachable_source_vertex()
                    .ok_or_else(|| GraphError::TopologyConflict)?
                    .geometry
                    .clone(),
            );
            let translation = half.lateral()? * distance;
            *vertices.0.as_position_mut() = vertices.0.as_position().clone() + translation.clone();
            *vertices.1.as_position_mut() = vertices.1.as_position().clone() + translation;
            (vertices, half.geometry.clone())
        };
        Ok(HalfExtrudeCache { ab, vertices, half })
    }
}

pub fn remove_with_cache<M, N, G>(
    mut mutation: N,
    cache: EdgeRemoveCache<G>,
) -> Result<(Half<G>, Half<G>), GraphError>
where
    N: AsMut<Mutation<M, G>>,
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: Geometry,
{
    fn remove<M, N, G>(mut mutation: N, cache: HalfRemoveCache<G>) -> Result<Half<G>, GraphError>
    where
        N: AsMut<Mutation<M, G>>,
        M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
        G: Geometry,
    {
        let HalfRemoveCache { ab, cache, .. } = cache;
        if let Some(cache) = cache {
            face::remove_with_cache(mutation.as_mut(), cache)?;
        }
        mutation
            .as_mut()
            .storage
            .remove(&ab)
            .ok_or_else(|| GraphError::TopologyNotFound)
    }

    let EdgeRemoveCache {
        a,
        b,
        half,
        opposite,
        ..
    } = cache;
    // Connect each vertex to a remaining outgoing edge.
    if let Some(ax) = opposite.bx {
        mutation.as_mut().connect_outgoing_half(a, ax)?;
    }
    if let Some(bx) = half.bx {
        mutation.as_mut().connect_outgoing_half(b, bx)?;
    }
    // Connect previous and next edges across the composite edge to be removed.
    if let (Some(xa), Some(ax)) = (half.xa, opposite.bx) {
        mutation.as_mut().connect_neighboring_halves(xa, ax)?;
    }
    if let (Some(xb), Some(bx)) = (opposite.xa, half.bx) {
        mutation.as_mut().connect_neighboring_halves(xb, bx)?;
    }
    Ok((
        remove(mutation.as_mut(), half)?,
        remove(mutation.as_mut(), opposite)?,
    ))
}

pub fn split_with_cache<M, N, G>(
    mut mutation: N,
    cache: EdgeSplitCache<G>,
) -> Result<VertexKey, GraphError>
where
    N: AsMut<Mutation<M, G>>,
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: Geometry,
{
    fn remove<M, N, G>(mut mutation: N, ab: HalfKey) -> Result<Half<G>, GraphError>
    where
        N: AsMut<Mutation<M, G>>,
        M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
        G: Geometry,
    {
        let (a, _) = ab.into();
        mutation.as_mut().disconnect_outgoing_half(a)?;
        let xa = mutation.as_mut().disconnect_previous_half(ab)?;
        let bx = mutation.as_mut().disconnect_next_half(ab)?;
        let mut half = mutation.as_mut().storage.remove(&ab).unwrap();
        // Restore the connectivity of the edge. The mutations will clear this
        // data, because it is still a part of the mesh at that point.
        half.previous = xa;
        half.next = bx;
        Ok(half)
    }

    fn split_at_vertex<M, N, G>(
        mut mutation: N,
        a: VertexKey,
        b: VertexKey,
        m: VertexKey,
        ab: HalfKey,
    ) -> Result<(HalfKey, HalfKey), GraphError>
    where
        N: AsMut<Mutation<M, G>>,
        M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
        G: Geometry,
    {
        // Remove the half-edge and insert two truncated edges in its place.
        let Half {
            next,
            previous,
            face,
            geometry,
            ..
        } = remove(mutation.as_mut(), ab)?;
        let am = mutation
            .as_mut()
            .get_or_insert_edge_with((a, m), || geometry.clone())
            .map(|(am, _)| am)?;
        let mb = mutation
            .as_mut()
            .get_or_insert_edge_with((m, b), move || geometry)
            .map(|(mb, _)| mb)?;
        // Connect the new half-edges to each other and their leading
        // half-edges.
        mutation.as_mut().connect_neighboring_halves(am, mb)?;
        if let Some(xa) = previous {
            mutation.as_mut().connect_neighboring_halves(xa, am)?;
        }
        if let Some(bx) = next {
            mutation.as_mut().connect_neighboring_halves(mb, bx)?;
        }
        // Update the associated face, if any, because it may refer to the
        // removed half-edge.
        if let Some(abc) = face {
            mutation.as_mut().connect_face_to_half(am, abc)?;
            mutation.as_mut().connect_half_to_face(am, abc)?;
            mutation.as_mut().connect_half_to_face(mb, abc)?;
        }
        Ok((am, mb))
    }

    let EdgeSplitCache {
        a,
        b,
        ab,
        ba,
        midpoint,
        ..
    } = cache;
    let m = mutation.as_mut().insert_vertex(midpoint);
    // Split the half-edges.
    split_at_vertex(mutation.as_mut(), a, b, m, ab)?;
    split_at_vertex(mutation.as_mut(), b, a, m, ba)?;
    Ok(m)
}

pub fn bridge_with_cache<M, N, G>(
    mut mutation: N,
    cache: HalfBridgeCache<G>,
) -> Result<FaceKey, GraphError>
where
    N: AsMut<Mutation<M, G>>,
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: Geometry,
{
    let HalfBridgeCache {
        a,
        b,
        c,
        d,
        half,
        face,
        ..
    } = cache;
    mutation.as_mut().insert_face(&[a, b, c, d], (half, face))
}

pub fn extrude_with_cache<M, N, G, T>(
    mut mutation: N,
    cache: HalfExtrudeCache<G>,
) -> Result<HalfKey, GraphError>
where
    N: AsMut<Mutation<M, G>>,
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: Geometry + EdgeLateral,
    G::Lateral: Mul<T>,
    G::Vertex: AsPosition,
    ScaledEdgeLateral<G, T>: Clone,
    VertexPosition<G>: Add<ScaledEdgeLateral<G, T>, Output = VertexPosition<G>> + Clone,
{
    let HalfExtrudeCache {
        ab, vertices, half, ..
    } = cache;
    let mutation = mutation.as_mut();
    let c = mutation.insert_vertex(vertices.0);
    let d = mutation.insert_vertex(vertices.1);
    // TODO: If this half-edge already exists, then this should probably return
    //       an error.
    let cd = mutation
        .get_or_insert_edge_with((c, d), move || half)
        .map(|(cd, _)| cd)?;
    let cache = HalfBridgeCache::snapshot(
        &Core::empty()
            .bind(mutation.as_vertex_storage())
            .bind(mutation.as_half_storage())
            .bind(mutation.as_face_storage()),
        ab,
        cd,
    )?;
    bridge_with_cache(mutation, cache).map(|_| cd)
}
