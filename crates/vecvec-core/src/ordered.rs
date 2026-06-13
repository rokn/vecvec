//! Total-ordering wrappers for floating-point scores.
//!
//! IEEE-754 floats are only `PartialOrd` (because `NaN` is unordered), which keeps
//! them out of `BinaryHeap`, `BTreeMap`, and `sort()`. Search ranks vectors by
//! distance/similarity scores constantly, so we need a total order. These wrappers
//! impose one via [`f32::total_cmp`] / [`f64::total_cmp`], which orders all values
//! including `NaN` and distinguishes `-0.0 < +0.0`.
//!
//! `Eq`/`Hash` are made consistent with that order by comparing/hashing the raw
//! bit pattern, so two `OrderedF32` are equal iff they compare `Equal`.

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

macro_rules! ordered_float {
    ($(#[$meta:meta])* $name:ident($inner:ty => $bits:ty)) => {
        $(#[$meta])*
        #[derive(Clone, Copy)]
        #[repr(transparent)]
        pub struct $name($inner);

        impl $name {
            /// Wraps a float in a totally-ordered float (`NaN` is permitted and sorts
            /// as the greatest value, per `total_cmp`).
            #[inline]
            pub const fn new(value: $inner) -> Self {
                Self(value)
            }

            /// Returns the wrapped float.
            #[inline]
            pub const fn into_inner(self) -> $inner {
                self.0
            }
        }

        impl PartialEq for $name {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.0.total_cmp(&other.0) == Ordering::Equal
            }
        }
        impl Eq for $name {}

        impl PartialOrd for $name {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for $name {
            #[inline]
            fn cmp(&self, other: &Self) -> Ordering {
                self.0.total_cmp(&other.0)
            }
        }

        impl Hash for $name {
            #[inline]
            fn hash<H: Hasher>(&self, state: &mut H) {
                // Consistent with the `total_cmp`-based `Eq`: distinct bit patterns
                // (e.g. -0.0 vs +0.0) are unequal and hash differently.
                self.0.to_bits().hash(state);
            }
        }

        impl From<$inner> for $name {
            #[inline]
            fn from(value: $inner) -> Self {
                Self(value)
            }
        }
        impl From<$name> for $inner {
            #[inline]
            fn from(v: $name) -> Self {
                v.0
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!(stringify!($name), "({})"), self.0)
            }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&self.0, f)
            }
        }

        // Silence "unused width param" lints in case `$bits` isn't referenced.
        const _: fn() = || {
            let _ = |x: $inner| -> $bits { x.to_bits() };
        };
    };
}

ordered_float!(
    /// A totally-ordered `f32`.
    OrderedF32(f32 => u32)
);
ordered_float!(
    /// A totally-ordered `f64`.
    OrderedF64(f64 => u64)
);

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BinaryHeap;

    #[test]
    fn sorts_like_total_cmp() {
        let mut v = [
            OrderedF32::new(1.5),
            OrderedF32::new(-2.0),
            OrderedF32::new(0.0),
            OrderedF32::new(f32::NAN),
            OrderedF32::new(f32::INFINITY),
        ];
        v.sort();
        // -2.0 < 0.0 < 1.5 < +inf < NaN  (NaN is greatest under total_cmp)
        assert_eq!(v[0], OrderedF32::new(-2.0));
        assert_eq!(v[1], OrderedF32::new(0.0));
        assert_eq!(v[2], OrderedF32::new(1.5));
        assert_eq!(v[3], OrderedF32::new(f32::INFINITY));
        assert!(v[4].into_inner().is_nan());
    }

    #[test]
    fn signed_zeroes_are_distinguished() {
        let neg = OrderedF32::new(-0.0);
        let pos = OrderedF32::new(0.0);
        assert_ne!(neg, pos);
        assert!(neg < pos);
    }

    #[test]
    fn usable_in_a_max_heap() {
        let mut heap: BinaryHeap<OrderedF64> =
            [3.0, 1.0, 2.0].into_iter().map(OrderedF64::new).collect();
        assert_eq!(heap.pop(), Some(OrderedF64::new(3.0)));
        assert_eq!(heap.pop(), Some(OrderedF64::new(2.0)));
    }

    #[test]
    fn eq_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(OrderedF32::new(2.0));
        assert!(set.contains(&OrderedF32::new(2.0)));
        assert!(!set.contains(&OrderedF32::new(-0.0)));
    }
}
