//! Generation-stamped visited sets, pooled for reuse across searches.
//!
//! A graph traversal must mark nodes visited. Allocating a fresh `bool` vector per
//! search is wasteful; instead each [`VisitedList`] keeps a generation counter and
//! a `Vec<u32>` of stamps — "clearing" just bumps the generation in O(1). A
//! [`VisitedPool`] hands these out so concurrent searches don't allocate.

use parking_lot::Mutex;

/// A reusable visited set sized to the number of points.
pub(crate) struct VisitedList {
    generation: u32,
    stamps: Vec<u32>,
}

impl VisitedList {
    /// Creates a visited set for `n` points.
    pub(crate) fn new(n: usize) -> Self {
        Self {
            generation: 1,
            stamps: vec![0; n],
        }
    }

    fn ensure_capacity(&mut self, n: usize) {
        if self.stamps.len() < n {
            self.stamps.resize(n, 0);
        }
    }

    /// Resets all marks in O(1) (amortized) by advancing the generation.
    pub(crate) fn clear(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            // Wrapped: do a real clear once, then resume.
            self.stamps.iter_mut().for_each(|s| *s = 0);
            self.generation = 1;
        }
    }

    /// Marks `i` visited, returning `true` if it was **not** already visited this
    /// generation.
    #[inline]
    pub(crate) fn visit(&mut self, i: u32) -> bool {
        let idx = i as usize;
        if self.stamps[idx] == self.generation {
            false
        } else {
            self.stamps[idx] = self.generation;
            true
        }
    }
}

/// A pool of [`VisitedList`]s for `n`-point graphs.
pub(crate) struct VisitedPool {
    n: usize,
    free: Mutex<Vec<VisitedList>>,
}

impl VisitedPool {
    /// Creates a pool for graphs of `n` points.
    pub(crate) fn new(n: usize) -> Self {
        Self {
            n,
            free: Mutex::new(Vec::new()),
        }
    }

    /// Checks out a cleared visited list (allocating only if the pool is empty).
    pub(crate) fn get(&self) -> VisitedList {
        if let Some(mut list) = self.free.lock().pop() {
            list.ensure_capacity(self.n);
            list.clear();
            list
        } else {
            VisitedList::new(self.n)
        }
    }

    /// Returns a visited list to the pool for reuse.
    pub(crate) fn put(&self, list: VisitedList) {
        self.free.lock().push(list);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visit_marks_once_per_generation() {
        let mut v = VisitedList::new(8);
        assert!(v.visit(3));
        assert!(!v.visit(3));
        assert!(v.visit(4));
        v.clear();
        assert!(v.visit(3)); // visible again after clear
    }

    #[test]
    fn pool_reuses_lists() {
        let pool = VisitedPool::new(16);
        let mut a = pool.get();
        assert!(a.visit(1));
        pool.put(a);
        let mut b = pool.get();
        assert!(b.visit(1)); // cleared on checkout
    }
}
