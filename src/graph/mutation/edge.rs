use failure::Error;
use std::ops::{Add, Deref, DerefMut, Mul};

use geometry::convert::AsPosition;
use geometry::Geometry;
use graph::geometry::alias::{ScaledEdgeLateral, VertexPosition};
use graph::geometry::{EdgeLateral, EdgeMidpoint};
use graph::mesh::Mesh;
use graph::mutation::vertex::VertexMutation;
use graph::mutation::{Commit, Mode, Mutate, Mutation};
use graph::storage::convert::AsStorage;
use graph::storage::{Bind, Core, EdgeKey, FaceKey, Storage, VertexKey};
use graph::topology::{Edge, Face, Vertex};
use graph::view::convert::IntoView;
use graph::view::{EdgeView, Inconsistent};
use graph::{GraphError, Perimeter, ResultExt};

pub struct EdgeMutation<G>
where
    G: Geometry,
{
    mutation: VertexMutation<G>,
    storage: Storage<Edge<G>>,
}

impl<G> EdgeMutation<G>
where
    G: Geometry,
{
    pub fn insert_edge(
        &mut self,
        vertices: (VertexKey, VertexKey),
        geometry: G::Edge,
    ) -> Result<EdgeKey, Error> {
        let (a, b) = vertices;
        let ab = (a, b).into();
        let ba = (b, a).into();
        // If the edge already exists, then fail. This ensures an important
        // invariant: edges may only have two adjacent faces. That is, a
        // half-edge may only have one associated face, at most one preceding
        // half-edge, at most one following half-edge, and may form at most one
        // closed loop.
        if self.storage.contains_key(&ab) {
            return Err(GraphError::TopologyConflict.into());
        }
        if !self.mutation.as_storage().contains_key(&b) {
            return Err(GraphError::TopologyNotFound.into());
        }
        let mut edge = Edge::new(b, geometry);
        if let Some(opposite) = self.storage.get_mut(&ba) {
            edge.opposite = Some(ba);
            opposite.opposite = Some(ab);
        }
        self.storage.insert_with_key(&ab, edge);
        self.connect_outgoing_edge(a, ab)?;
        Ok(ab)
    }

    pub fn get_or_insert_edge(
        &mut self,
        vertices: (VertexKey, VertexKey),
        geometry: G::Edge,
    ) -> Result<EdgeKey, Error> {
        self.insert_edge(vertices, geometry)
            .or_if_conflict(|| Ok(vertices.into()))
    }

    pub fn get_or_insert_composite_edge(
        &mut self,
        vertices: (VertexKey, VertexKey),
        geometry: G::Edge,
    ) -> Result<(EdgeKey, EdgeKey), Error> {
        let (a, b) = vertices;
        let ab = self.get_or_insert_edge((a, b), geometry.clone())?;
        let ba = self.get_or_insert_edge((b, a), geometry)?;
        Ok((ab, ba))
    }

    pub fn remove_edge(&mut self, ab: EdgeKey) -> Result<Edge<G>, Error> {
        let (a, _) = ab.to_vertex_keys();
        self.disconnect_next_edge(ab)?;
        self.disconnect_previous_edge(ab)?;
        self.disconnect_outgoing_edge(a)?;
        Ok(self.storage.remove(&ab).unwrap())
    }

    pub fn remove_composite_edge(&mut self, ab: EdgeKey) -> Result<(Edge<G>, Edge<G>), Error> {
        let (a, b) = ab.to_vertex_keys();
        let edge = self.remove_edge((a, b).into())?;
        let opposite = self.remove_edge((b, a).into())?;
        Ok((edge, opposite))
    }

    pub fn connect_neighboring_edges(&mut self, ab: EdgeKey, bc: EdgeKey) -> Result<(), Error> {
        if ab.to_vertex_keys().1 == bc.to_vertex_keys().0 {
            let previous = match self.storage.get_mut(&ab) {
                Some(previous) => {
                    previous.next = Some(bc);
                    Ok(())
                }
                _ => Err(Error::from(GraphError::TopologyNotFound)),
            };
            let next = match self.storage.get_mut(&bc) {
                Some(next) => {
                    next.previous = Some(ab);
                    Ok(())
                }
                _ => Err(Error::from(GraphError::TopologyNotFound)),
            };
            previous.and(next)
        }
        else {
            Err(Error::from(GraphError::TopologyMalformed))
        }
    }

    pub fn disconnect_next_edge(&mut self, ab: EdgeKey) -> Result<Option<EdgeKey>, Error> {
        let bx = {
            self.storage
                .get_mut(&ab)
                .ok_or_else(|| Error::from(GraphError::TopologyNotFound))?
                .next
                .take()
        };
        if let Some(bx) = bx.as_ref() {
            self.storage.get_mut(bx).unwrap().previous = None;
        }
        Ok(bx)
    }

    pub fn disconnect_previous_edge(&mut self, ab: EdgeKey) -> Result<Option<EdgeKey>, Error> {
        let xa = {
            self.storage
                .get_mut(&ab)
                .ok_or_else(|| Error::from(GraphError::TopologyNotFound))?
                .previous
                .take()
        };
        if let Some(xa) = xa.as_ref() {
            self.storage.get_mut(xa).unwrap().previous = None;
        }
        Ok(xa)
    }

    pub fn connect_edge_to_face(&mut self, ab: EdgeKey, abc: FaceKey) -> Result<(), Error> {
        let edge = self
            .storage
            .get_mut(&ab)
            .ok_or_else(|| Error::from(GraphError::TopologyNotFound))?;
        edge.face = Some(abc);
        Ok(())
    }

    pub fn disconnect_edge_from_face(&mut self, ab: EdgeKey) -> Result<Option<FaceKey>, Error> {
        let face = self
            .storage
            .get_mut(&ab)
            .ok_or_else(|| Error::from(GraphError::TopologyNotFound))?
            .face
            .take();
        Ok(face)
    }
}

impl<G> AsStorage<Edge<G>> for EdgeMutation<G>
where
    G: Geometry,
{
    fn as_storage(&self) -> &Storage<Edge<G>> {
        &self.storage
    }
}

impl<G> Commit<G> for EdgeMutation<G>
where
    G: Geometry,
{
    type Error = Error;

    fn commit(self) -> Result<Self::Mutant, Self::Error> {
        let EdgeMutation {
            mutation,
            storage: edges,
            ..
        } = self;
        mutation
            .commit()
            .and_then(move |vertices| Ok((vertices, edges)))
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

impl<G> Mode<G> for EdgeMutation<G>
where
    G: Geometry,
{
    type Mutant = (Storage<Vertex<G>>, Storage<Edge<G>>);
}

impl<G> Mutate<G> for EdgeMutation<G>
where
    G: Geometry,
{
    fn mutate(mutant: Self::Mutant) -> Self {
        let (vertices, edges) = mutant;
        EdgeMutation {
            mutation: VertexMutation::mutate(vertices),
            storage: edges,
        }
    }
}

pub struct EdgeSplitCache<G>
where
    G: Geometry,
{
    ab: EdgeKey,
    ba: EdgeKey,
    midpoint: G::Vertex,
}

impl<G> EdgeSplitCache<G>
where
    G: EdgeMidpoint<Midpoint = VertexPosition<G>> + Geometry,
    G::Vertex: AsPosition,
{
    pub fn snapshot<M>(storage: M, ab: EdgeKey) -> Result<Self, Error>
    where
        M: AsStorage<Edge<G>> + AsStorage<Vertex<G>>,
    {
        let (a, b) = ab.to_vertex_keys();
        let edge: EdgeView<M, G, _> = match (ab, storage).into_view() {
            Some(edge) => edge,
            _ => return Err(GraphError::TopologyNotFound.into()),
        };
        let mut midpoint = edge.source_vertex().geometry.clone();
        *midpoint.as_position_mut() = EdgeMidpoint::midpoint(&edge)?;
        Ok(EdgeSplitCache {
            ab,
            ba: (b, a).into(),
            midpoint,
        })
    }
}

pub struct EdgeJoinCache<G>
where
    G: Geometry,
{
    source: EdgeKey,
    destination: EdgeKey,
    edge: G::Edge,
    face: G::Face,
}

impl<G> EdgeJoinCache<G>
where
    G: Geometry,
{
    pub fn snapshot<M>(storage: M, source: EdgeKey, destination: EdgeKey) -> Result<Self, Error>
    where
        M: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>>,
    {
        let (a, b) = source.to_vertex_keys();
        let (c, d) = destination.to_vertex_keys();
        let source = match (
            EdgeView::<_, _, Inconsistent>::from_keyed_storage(source, &storage),
            EdgeView::<_, _, Inconsistent>::from_keyed_storage(destination, &storage),
        ) {
            (Some(source), Some(_)) => source,
            _ => return Err(GraphError::TopologyNotFound.into()),
        };
        // At this point, we can assume the points a, b, c, and d exist in the
        // mesh. Before mutating the mesh, ensure that existing interior edges
        // are boundaries.
        for edge in [a, b, c, d]
            .perimeter()
            .flat_map(|ab| EdgeView::<_, _, Inconsistent>::from_keyed_storage(ab.into(), &storage))
        {
            if !edge.is_boundary_edge() {
                return Err(GraphError::TopologyConflict.into());
            }
        }
        Ok(EdgeJoinCache {
            source: source.key(),
            destination,
            edge: source.geometry.clone(),
            face: source
                .reachable_opposite_edge()
                .and_then(|opposite| opposite.into_reachable_face())
                .map(|face| face.geometry.clone())
                .unwrap_or_else(Default::default),
        })
    }
}

pub struct EdgeExtrudeCache<G>
where
    G: Geometry,
{
    ab: EdgeKey,
    vertices: (G::Vertex, G::Vertex),
    edge: G::Edge,
}

impl<G> EdgeExtrudeCache<G>
where
    G: Geometry,
{
    pub fn snapshot<T>(storage: &Mesh<G>, ab: EdgeKey, distance: T) -> Result<Self, Error>
    where
        G: Geometry + EdgeLateral,
        G::Lateral: Mul<T>,
        G::Vertex: AsPosition,
        ScaledEdgeLateral<G, T>: Clone,
        VertexPosition<G>: Add<ScaledEdgeLateral<G, T>, Output = VertexPosition<G>> + Clone,
    {
        // Get the extruded geometry.
        let (vertices, edge) = {
            let edge = match storage.edge(ab) {
                Some(edge) => edge,
                _ => return Err(GraphError::TopologyNotFound.into()),
            };
            if !edge.is_boundary_edge() {
                return Err(GraphError::TopologyConflict.into());
            }
            let mut vertices = (
                edge.destination_vertex().geometry.clone(),
                edge.source_vertex().geometry.clone(),
            );
            let translation = edge.lateral()? * distance;
            *vertices.0.as_position_mut() = vertices.0.as_position().clone() + translation.clone();
            *vertices.1.as_position_mut() = vertices.1.as_position().clone() + translation;
            (vertices, edge.geometry.clone())
        };
        Ok(EdgeExtrudeCache { ab, vertices, edge })
    }
}

pub fn split_with_cache<G>(
    mutation: &mut Mutation<G>,
    cache: EdgeSplitCache<G>,
) -> Result<VertexKey, Error>
where
    G: EdgeMidpoint<Midpoint = VertexPosition<G>> + Geometry,
    G::Vertex: AsPosition,
{
    fn split_at_vertex<G>(
        mutation: &mut Mutation<G>,
        ab: EdgeKey,
        m: VertexKey,
    ) -> Result<(EdgeKey, EdgeKey), Error>
    where
        G: EdgeMidpoint<Midpoint = VertexPosition<G>> + Geometry,
        G::Vertex: AsPosition,
    {
        // Remove the edge and insert two truncated edges in its place.
        let (a, b) = ab.to_vertex_keys();
        let span = mutation.remove_edge(ab)?;
        let am = mutation.insert_edge((a, m), span.geometry.clone())?;
        let mb = mutation.insert_edge((m, b), span.geometry.clone())?;
        // Connect the new edges to each other and their leading edges.
        mutation.connect_neighboring_edges(am, mb)?;
        if let Some(xa) = span.previous {
            mutation.connect_neighboring_edges(xa, am)?;
        }
        if let Some(bx) = span.next {
            mutation.connect_neighboring_edges(mb, bx)?;
        }
        // Update the associated face, if any, because it may refer to the
        // removed edge.
        if let Some(abc) = span.face {
            mutation.connect_face_to_edge(am, abc)?;
            mutation.connect_edge_to_face(am, abc)?;
            mutation.connect_edge_to_face(mb, abc)?;
        }
        Ok((am, mb))
    }

    let EdgeSplitCache { ab, ba, midpoint } = cache;
    let m = mutation.insert_vertex(midpoint);
    // Split the half-edges.
    split_at_vertex(mutation, ab, m)?;
    split_at_vertex(mutation, ba, m)?;
    Ok(m)
}

pub fn join_with_cache<G>(
    mutation: &mut Mutation<G>,
    cache: EdgeJoinCache<G>,
) -> Result<EdgeKey, Error>
where
    G: Geometry,
{
    let EdgeJoinCache {
        source,
        destination,
        edge,
        face,
    } = cache;
    let (a, b) = source.to_vertex_keys();
    let (c, d) = destination.to_vertex_keys();
    // TODO: Split the face to form triangles.
    mutation.insert_face(&[a, b, c, d], (edge, face))?;
    Ok(source)
}

pub fn extrude_with_cache<G, T>(
    mutation: &mut Mutation<G>,
    cache: EdgeExtrudeCache<G>,
) -> Result<EdgeKey, Error>
where
    G: Geometry + EdgeLateral,
    G::Lateral: Mul<T>,
    G::Vertex: AsPosition,
    ScaledEdgeLateral<G, T>: Clone,
    VertexPosition<G>: Add<ScaledEdgeLateral<G, T>, Output = VertexPosition<G>> + Clone,
{
    let EdgeExtrudeCache { ab, vertices, edge } = cache;
    let c = mutation.insert_vertex(vertices.0);
    let d = mutation.insert_vertex(vertices.1);
    let cd = mutation.insert_edge((c, d), edge)?;
    let cache = EdgeJoinCache::snapshot(
        Core::empty()
            .bind(AsStorage::<Vertex<G>>::as_storage(mutation))
            .bind(AsStorage::<Edge<G>>::as_storage(mutation))
            .bind(AsStorage::<Face<G>>::as_storage(mutation)),
        ab,
        cd,
    )?;
    join_with_cache(mutation, cache)
}
