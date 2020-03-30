use std::collections::HashSet;
use std::hash::Hash;

/// Trace of a path.
pub trait Trace<T> {
    /// Inserts the given breadcrumb into the trace.
    ///
    /// If an intersection with the trace is detected, then this function
    /// returns `false` and otherwise returns `true` (similarly to collections
    /// like `HashSet). If `false` is returned, then the iteration should
    /// terminate.
    fn insert(&mut self, breadcrumb: T) -> bool;
}

/// Trace that detects the first breadcrumb that is encountered.
///
/// This trace only stores the first breadcrumb in a traversal and should
/// **not** be used when traversing a graph with unknown consistency, because it
/// may never signal that the iteration should terminate. However, it requires
/// very little space and time to operate.
#[derive(Clone, Copy, Debug, Default)]
pub struct TraceFirst<T>
where
    T: Copy,
{
    breadcrumb: Option<T>,
}

impl<T> Trace<T> for TraceFirst<T>
where
    T: Copy + Eq,
{
    fn insert(&mut self, breadcrumb: T) -> bool {
        match self.breadcrumb {
            Some(intersection) => intersection != breadcrumb,
            None => {
                self.breadcrumb = Some(breadcrumb);
                true
            }
        }
    }
}

/// Trace that detects any breadcrumb that has been previously encountered.
///
/// This trace stores all breadcrumbs and detects any and all collisions. This
/// is very robust, but requires space for breadcrumbs and must hash breadcrumbs
/// to detect collisions.
#[derive(Clone, Debug, Default)]
pub struct TraceAny<T>
where
    T: Copy + Eq + Hash,
{
    breadcrumbs: HashSet<T>,
}

impl<T> Trace<T> for TraceAny<T>
where
    T: Copy + Eq + Hash,
{
    fn insert(&mut self, breadcrumb: T) -> bool {
        self.breadcrumbs.insert(breadcrumb)
    }
}
