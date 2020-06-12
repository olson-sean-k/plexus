// TODO: The `derivative` attribute macro causes this lint failure.
#![allow(clippy::match_single_binding)]

use derivative::Derivative;
use num::{One, Zero};
use std::cmp::Reverse;
use std::collections::hash_map::Entry;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::entity::storage::AsStorage;
use crate::entity::traverse::Adjacency;
use crate::entity::view::{Bind, Unbind};

pub trait Metric: Eq + One + Ord + Zero {}

impl<Q> Metric for Q where Q: Eq + One + Ord + Zero {}

#[derivative(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
#[derive(Derivative)]
struct KeyedMetric<K, Q>(
    #[derivative(Ord = "ignore", PartialEq = "ignore", PartialOrd = "ignore")] K,
    Q,
)
where
    Q: Eq + Ord;

// TODO: This may become part of the public API.
#[allow(dead_code)]
pub fn metrics<'a, M, T>(from: T, to: Option<T::Key>) -> HashMap<T::Key, (Option<T::Key>, usize)>
where
    M: 'a + AsStorage<T::Entity>,
    T: Adjacency + Bind<&'a M> + Copy + Unbind<&'a M>,
{
    metrics_with(from, to, |_, _| 1usize)
}

pub fn metrics_with<'a, M, T, Q, F>(
    from: T,
    to: Option<T::Key>,
    f: F,
) -> HashMap<T::Key, (Option<T::Key>, Q)>
where
    M: 'a + AsStorage<T::Entity>,
    T: Adjacency + Bind<&'a M> + Copy + Unbind<&'a M>,
    Q: Copy + Metric,
    F: Fn(T, T) -> Q,
{
    let (storage, from) = from.unbind();
    let capacity = if to.is_some() {
        0
    }
    else {
        storage.as_storage().len()
    };
    let mut buffer = BinaryHeap::new();
    let mut breadcrumbs = HashSet::with_capacity(capacity);
    let mut metrics = HashMap::with_capacity(capacity);

    metrics.insert(from, (None, Q::zero()));
    buffer.push(KeyedMetric(from, Reverse(Q::zero())));
    while let Some(KeyedMetric(key, Reverse(metric))) = buffer.pop() {
        if Some(key) == to {
            break;
        }
        let entity = T::bind(storage, key).unwrap();
        if breadcrumbs.insert(entity.key()) {
            for adjacent in entity
                .adjacency()
                .into_iter()
                .map(|key| T::bind(storage, key).unwrap())
            {
                // TODO: Consider returning an error if the output of `f` is
                //       less than zero.
                let metric = metric + f(entity, adjacent);
                match metrics.entry(adjacent.key()) {
                    Entry::Occupied(entry) => {
                        if metric < entry.get().1 {
                            entry.into_mut().1 = metric;
                        }
                    }
                    Entry::Vacant(entry) => {
                        entry.insert((Some(entity.key()), metric));
                    }
                }
                buffer.push(KeyedMetric(adjacent.key(), Reverse(metric)));
            }
        }
    }
    metrics
}

#[cfg(test)]
mod tests {
    use decorum::R64;
    use nalgebra::Point2;
    use theon::space::InnerSpace;

    use crate::entity::dijkstra;
    use crate::graph::MeshGraph;
    use crate::prelude::*;
    use crate::primitive::{Tetragon, Trigon};

    #[test]
    fn logical_metrics() {
        let graph = MeshGraph::<()>::from_raw_buffers(vec![Trigon::new(0usize, 1, 2)], vec![(); 3])
            .unwrap();
        let vertex = graph.vertices().nth(0).unwrap();
        let metrics = dijkstra::metrics(vertex, None);
        let a = vertex.key();
        let b = vertex.outgoing_arc().destination_vertex().key();
        let c = vertex.outgoing_arc().next_arc().destination_vertex().key();
        let aq = *metrics.get(&a).unwrap();
        let bq = *metrics.get(&b).unwrap();
        let cq = *metrics.get(&c).unwrap();

        assert_eq!(aq, (None, 0));
        assert_eq!(bq, (Some(a), 1));
        assert_eq!(cq, (Some(a), 1));
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn euclidean_distance_metrics() {
        let graph = MeshGraph::<Point2<f64>>::from_raw_buffers(
            vec![Tetragon::new(0usize, 1, 2, 3)],
            vec![(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
        )
        .unwrap();
        let vertex = graph.vertices().nth(0).unwrap();
        let metrics = dijkstra::metrics_with(vertex, None, |from, to| {
            R64::from((to.position() - from.position()).magnitude())
        });
        let a = vertex.key();
        let b = vertex.outgoing_arc().destination_vertex().key();
        let c = vertex.outgoing_arc().next_arc().destination_vertex().key();
        let d = vertex
            .outgoing_arc()
            .next_arc()
            .next_arc()
            .destination_vertex()
            .key();
        let aq = *metrics.get(&a).unwrap();
        let bq = *metrics.get(&b).unwrap();
        let cq = *metrics.get(&c).unwrap();
        let dq = *metrics.get(&d).unwrap();

        assert_eq!(aq, (None, 0.0.into()));
        assert_eq!(bq, (Some(a), 2.0.into()));
        assert_eq!(cq, (Some(b), 4.0.into()));
        assert_eq!(dq, (Some(a), 2.0.into()));
    }
}
