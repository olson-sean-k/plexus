use std::cmp::{Ord, Ordering, PartialOrd};
use std::collections::BinaryHeap;
use std::iter::FromIterator;
use theon::space::{EuclideanSpace, FiniteDimensional, Scalar};
use theon::{AsPosition, Position};
use typenum::U2;

use crate::primitive::decompose::IntoEdges;
use crate::primitive::{Edge, Polygonal};

#[derive(Clone, Copy, Debug)]
struct XMajor<G>
where
    G: AsPosition,
    Position<G>: FiniteDimensional<N = U2>,
{
    pub geometry: G,
}

impl<G> Eq for XMajor<G>
where
    G: AsPosition,
    Position<G>: Eq + FiniteDimensional<N = U2>,
{
}

impl<G> From<G> for XMajor<G>
where
    G: AsPosition,
    Position<G>: FiniteDimensional<N = U2>,
{
    fn from(geometry: G) -> Self {
        XMajor { geometry }
    }
}

impl<G> Ord for XMajor<G>
where
    G: AsPosition,
    Position<G>: Eq + FiniteDimensional<N = U2>,
    Scalar<Position<G>>: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<G> PartialEq for XMajor<G>
where
    G: AsPosition,
    Position<G>: FiniteDimensional<N = U2> + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.geometry.as_position().eq(other.geometry.as_position())
    }
}

impl<G> PartialOrd for XMajor<G>
where
    G: AsPosition,
    Position<G>: FiniteDimensional<N = U2>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let (ax, ay) = self.geometry.as_position().into_xy();
        let (bx, by) = other.geometry.as_position().into_xy();
        match (ax.partial_cmp(&bx), ay.partial_cmp(&by)) {
            (Some(Ordering::Equal), Some(ordering)) => Some(ordering),
            (Some(ordering), Some(_)) => Some(ordering),
            _ => None,
        }
    }
}

// TODO: It should be possible to use this to implement both the Bentley-
//       Ottmann and Shamos-Hoey algorithms. Note that both start with an event
//       queue initialized from edges (segments) in the same way.
pub struct SweepLine<G>
where
    G: AsPosition,
    Position<G>: FiniteDimensional<N = U2>,
{
    edges: Vec<Edge<G>>,
    events: BinaryHeap<XMajor<PointEvent<G>>>,
}

impl<G> SweepLine<G>
where
    G: AsPosition,
    Position<G>: Eq + FiniteDimensional<N = U2>,
    Scalar<Position<G>>: Ord,
{
    pub fn intersections(self) -> Vec<Position<G>> {
        let SweepLine { edges, mut events } = self;
        let mut intersections = vec![];
        while let Some(XMajor { geometry: event }) = events.pop() {
            match event {
                PointEvent::Start { geometry, index } => {}
                PointEvent::End { geometry, index } => {}
                PointEvent::Intersection { position, .. } => {
                    intersections.push(position);
                }
            }
        }
        intersections
    }
}

impl<P, G> From<P> for SweepLine<G>
where
    P: IntoEdges + Polygonal<Vertex = G>,
    G: AsPosition + Copy,
    Position<G>: Eq + FiniteDimensional<N = U2>,
    Scalar<Position<G>>: Ord,
{
    fn from(polygon: P) -> Self {
        polygon.into_edges().into_iter().collect()
    }
}

impl<G> FromIterator<Edge<G>> for SweepLine<G>
where
    G: AsPosition + Copy,
    Position<G>: Eq + FiniteDimensional<N = U2>,
    Scalar<Position<G>>: Ord,
{
    fn from_iter<I>(edges: I) -> Self
    where
        I: IntoIterator<Item = Edge<G>>,
    {
        let edges: Vec<_> = edges.into_iter().collect();
        let mut events = BinaryHeap::with_capacity(edges.len() * 2);
        for (index, edge) in edges.iter().enumerate() {
            let (start, end) = PointEvent::from_edge(edge, index);
            events.push(start.into());
            events.push(end.into());
        }
        SweepLine { edges, events }
    }
}

// "Start" and "end" refer to endpoints nearest the start and end of a sweep of
// a point, line, or plane. The direction of such a sweep is arbitrary; these
// terms are relative to that direction.
#[derive(Clone, Copy)]
enum PointEvent<G>
where
    G: AsPosition,
{
    Start {
        geometry: G,
        index: usize,
    },
    End {
        geometry: G,
        index: usize,
    },
    Intersection {
        position: Position<G>,
        above: usize,
        below: usize,
    },
}

impl<G> PointEvent<G>
where
    G: AsPosition + Copy,
    Position<G>: Eq + FiniteDimensional<N = U2>,
    Scalar<Position<G>>: Ord,
{
    pub fn from_edge(edge: &Edge<G>, index: usize) -> (Self, Self) {
        let a = XMajor::from(edge[0]);
        let b = XMajor::from(edge[1]);
        let (start, end) = if a < b { (a, b) } else { (b, a) };
        (
            PointEvent::Start {
                geometry: start.geometry,
                index,
            },
            PointEvent::End {
                geometry: end.geometry,
                index,
            },
        )
    }
}

// This is needed to sort `PointEvent`s by embedding them within `XMajor`s.
impl<G> AsPosition for PointEvent<G>
where
    G: AsPosition,
    G::Position: EuclideanSpace,
{
    type Position = G::Position;

    fn as_position(&self) -> &Self::Position {
        match self {
            PointEvent::Start { ref geometry, .. } => geometry.as_position(),
            PointEvent::End { ref geometry, .. } => geometry.as_position(),
            PointEvent::Intersection { ref position, .. } => position,
        }
    }
}
