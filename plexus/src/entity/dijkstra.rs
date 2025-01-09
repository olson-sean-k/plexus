use derivative::Derivative;
use std::cmp::Reverse;
use std::collections::hash_map::Entry;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::entity::storage::{AsStorage, Enumerate, Get};
use crate::entity::traverse::Adjacency;
use crate::entity::view::{Bind, Unbind};
use crate::entity::EntityError;
use crate::geometry::Metric;

pub type MetricTree<K, Q> = HashMap<K, (Option<K>, Q)>;

#[derive(Derivative)]
#[derivative(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
struct KeyedMetric<K, Q>(
    #[derivative(Ord = "ignore", PartialEq = "ignore", PartialOrd = "ignore")] K,
    Q,
)
where
    Q: Eq + Ord;

pub fn metrics_with<'a, M, T, Q, F>(
    from: T,
    to: Option<T::Key>,
    f: F,
) -> Result<MetricTree<T::Key, Q>, EntityError>
where
    M: 'a + AsStorage<T::Entity>,
    T: Adjacency + Bind<&'a M> + Copy + Unbind<&'a M>,
    Q: Copy + Metric,
    F: Fn(T, T) -> Q,
{
    let (storage, from) = from.unbind();
    let capacity = if let Some(key) = to {
        if !storage.as_storage().contains_key(&key) {
            return Err(EntityError::EntityNotFound);
        }
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
        let entity = T::bind(storage, key).ok_or(EntityError::EntityNotFound)?;
        if breadcrumbs.insert(entity.key()) {
            for adjacent in entity
                .adjacency()
                .into_iter()
                .map(|key| T::bind(storage, key))
            {
                let adjacent = adjacent.ok_or(EntityError::EntityNotFound)?;
                let summand = f(entity, adjacent);
                if summand < Q::zero() {
                    return Err(EntityError::Data);
                }
                let metric = metric + summand;
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
    Ok(metrics)
}

#[cfg(test)]
mod tests {
    use decorum::R64;
    use nalgebra::Point2;
    use theon::space::InnerSpace;

    use crate::entity::dijkstra::{self, MetricTree};
    use crate::entity::EntityError;
    use crate::graph::MeshGraph;
    use crate::prelude::*;
    use crate::primitive::{Tetragon, Trigon};

    #[test]
    fn decreasing_summand() {
        let graph = MeshGraph::<Point2<f64>>::from_raw_buffers(
            vec![Tetragon::new(0usize, 1, 2, 3)],
            vec![(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
        )
        .unwrap();
        let vertex = graph.vertices().next().unwrap();
        assert_eq!(
            Err(EntityError::Data),
            dijkstra::metrics_with(vertex, None, |_, _| -1isize)
        )
    }

    #[test]
    fn logical_metrics() {
        let graph = MeshGraph::<()>::from_raw_buffers(vec![Trigon::new(0usize, 1, 2)], vec![(); 3])
            .unwrap();
        let vertex = graph.vertices().next().unwrap();
        let metrics = dijkstra::metrics_with(vertex, None, |_, _| 1usize).unwrap();
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

    // TODO: Use approximated comparisons in assertions.
    #[test]
    fn euclidean_distance_metrics() {
        let graph = MeshGraph::<Point2<f64>>::from_raw_buffers(
            vec![Tetragon::new(0usize, 1, 2, 3)],
            vec![(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
        )
        .unwrap();
        let vertex = graph.vertices().next().unwrap();
        let metrics: MetricTree<_, R64> = dijkstra::metrics_with(vertex, None, |from, to| {
            R64::assert((to.position() - from.position()).magnitude())
        })
        .unwrap();
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

        assert_eq!(aq, (None, R64::assert(0.0)));
        assert_eq!(bq, (Some(a), R64::assert(2.0)));
        assert_eq!(cq, (Some(b), R64::assert(4.0)));
        assert_eq!(dq, (Some(a), R64::assert(2.0)));
    }
}
