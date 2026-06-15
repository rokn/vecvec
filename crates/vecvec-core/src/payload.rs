//! Per-point metadata (payload) and the filter DSL.
//!
//! A payload is a free-form JSON object attached to a point. Payloads live at the
//! **collection level** (a map from global id), like deletions, so segments stay
//! pure vectors. A [`Filter`] is a Qdrant-style boolean combination of field
//! conditions (`must` / `should` / `must_not`); it is evaluated per candidate during
//! search via [`FilterQuery`], which the segment adapts to the index's
//! [`FilterContext`](crate::index::FilterContext).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A point's metadata: a JSON value (expected to be an object).
pub type Payload = serde_json::Value;

/// The collection's payloads, keyed by global id.
pub type PayloadMap = HashMap<u64, Payload>;

/// A numeric range (any bound may be open).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Range {
    /// Strictly greater than.
    pub gt: Option<f64>,
    /// Greater than or equal.
    pub gte: Option<f64>,
    /// Strictly less than.
    pub lt: Option<f64>,
    /// Less than or equal.
    pub lte: Option<f64>,
}

impl Range {
    fn contains(&self, n: f64) -> bool {
        self.gt.is_none_or(|b| n > b)
            && self.gte.is_none_or(|b| n >= b)
            && self.lt.is_none_or(|b| n < b)
            && self.lte.is_none_or(|b| n <= b)
    }
}

/// A condition on a single payload field.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Condition {
    /// The field name (top-level key).
    pub key: String,
    /// Exact-match value (equality).
    #[serde(default, rename = "match")]
    pub r#match: Option<serde_json::Value>,
    /// Numeric range.
    #[serde(default)]
    pub range: Option<Range>,
}

impl Condition {
    fn eval(&self, payload: Option<&Payload>) -> bool {
        let value = payload.and_then(|p| p.get(&self.key));
        if let Some(expected) = &self.r#match
            && value != Some(expected)
        {
            return false;
        }
        if let Some(range) = &self.range {
            match value.and_then(serde_json::Value::as_f64) {
                Some(n) if range.contains(n) => {}
                _ => return false,
            }
        }
        true
    }
}

/// A boolean combination of conditions.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Filter {
    /// All must match (AND).
    #[serde(default)]
    pub must: Vec<Condition>,
    /// At least one must match if non-empty (OR).
    #[serde(default)]
    pub should: Vec<Condition>,
    /// None may match (NOT).
    #[serde(default)]
    pub must_not: Vec<Condition>,
}

impl Filter {
    /// Whether the filter has no conditions (matches everything).
    pub fn is_empty(&self) -> bool {
        self.must.is_empty() && self.should.is_empty() && self.must_not.is_empty()
    }

    /// Evaluates the filter against an (optional) payload.
    pub fn eval(&self, payload: Option<&Payload>) -> bool {
        if self.must.iter().any(|c| !c.eval(payload)) {
            return false;
        }
        if self.must_not.iter().any(|c| c.eval(payload)) {
            return false;
        }
        if !self.should.is_empty() && !self.should.iter().any(|c| c.eval(payload)) {
            return false;
        }
        true
    }
}

/// A filter plus the payload map to evaluate it against (passed into segment search).
#[derive(Clone, Copy)]
pub struct FilterQuery<'a> {
    /// The filter.
    pub filter: &'a Filter,
    /// The collection's payloads.
    pub payloads: &'a PayloadMap,
}

impl FilterQuery<'_> {
    /// Whether the point with global id `global` passes the filter.
    #[inline]
    pub fn matches(&self, global: u64) -> bool {
        self.filter.eval(self.payloads.get(&global))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn payload(v: serde_json::Value) -> PayloadMap {
        let mut m = PayloadMap::new();
        m.insert(1, v);
        m
    }

    #[test]
    fn match_and_range() {
        let payloads = payload(json!({"color": "red", "price": 12.5}));
        let red = Filter {
            must: vec![Condition {
                key: "color".into(),
                r#match: Some(json!("red")),
                range: None,
            }],
            ..Default::default()
        };
        assert!(
            FilterQuery {
                filter: &red,
                payloads: &payloads
            }
            .matches(1)
        );

        let cheap = Filter {
            must: vec![Condition {
                key: "price".into(),
                r#match: None,
                range: Some(Range {
                    lt: Some(20.0),
                    gte: Some(10.0),
                    ..Default::default()
                }),
            }],
            ..Default::default()
        };
        assert!(
            FilterQuery {
                filter: &cheap,
                payloads: &payloads
            }
            .matches(1)
        );

        let expensive = Filter {
            must: vec![Condition {
                key: "price".into(),
                r#match: None,
                range: Some(Range {
                    gt: Some(100.0),
                    ..Default::default()
                }),
            }],
            ..Default::default()
        };
        assert!(
            !FilterQuery {
                filter: &expensive,
                payloads: &payloads
            }
            .matches(1)
        );
    }

    #[test]
    fn should_and_must_not() {
        let payloads = payload(json!({"tag": "b"}));
        let f = Filter {
            should: vec![
                Condition {
                    key: "tag".into(),
                    r#match: Some(json!("a")),
                    range: None,
                },
                Condition {
                    key: "tag".into(),
                    r#match: Some(json!("b")),
                    range: None,
                },
            ],
            must_not: vec![Condition {
                key: "tag".into(),
                r#match: Some(json!("c")),
                range: None,
            }],
            ..Default::default()
        };
        assert!(
            FilterQuery {
                filter: &f,
                payloads: &payloads
            }
            .matches(1)
        );
        // A point with no payload (id 99) fails the `should`.
        assert!(
            !FilterQuery {
                filter: &f,
                payloads: &payloads
            }
            .matches(99)
        );
    }

    #[test]
    fn match_int_vs_float_equality() {
        // `Condition::eval` does a raw serde_json `Value` comparison, and serde_json
        // treats Number(3) and Number(3.0) as DISTINCT. So `match` is type-sensitive:
        // an int filter only matches an int-stored value (and float only float).
        // Crossing the types silently drops every result — pin that contract here.
        let int_stored = payload(json!({ "n": 3 }));
        let float_stored = payload(json!({ "n": 3.0 }));

        let cond = |val: serde_json::Value| Filter {
            must: vec![Condition {
                key: "n".into(),
                r#match: Some(val),
                range: None,
            }],
            ..Default::default()
        };
        let matches = |f: &Filter, p: &PayloadMap| {
            FilterQuery {
                filter: f,
                payloads: p,
            }
            .matches(1)
        };

        // Same type on both sides matches.
        assert!(matches(&cond(json!(3)), &int_stored));
        assert!(matches(&cond(json!(3.0)), &float_stored));
        // Crossing int/float does NOT match (documents the typing footgun).
        assert!(!matches(&cond(json!(3.0)), &int_stored));
        assert!(!matches(&cond(json!(3)), &float_stored));
    }

    #[test]
    fn must_not_does_not_exclude_missing_field() {
        // A missing field reads as `None`, so a `match` on an absent field is false,
        // and `must_not` over a false condition does NOT exclude the point. (must_not
        // is true negation, not a presence filter.)
        let payloads = payload(json!({ "tag": "b" }));
        let f = Filter {
            must_not: vec![Condition {
                key: "other".into(),
                r#match: Some(json!("x")),
                range: None,
            }],
            ..Default::default()
        };
        // Point lacks "other" -> must_not doesn't fire -> passes.
        assert!(
            FilterQuery {
                filter: &f,
                payloads: &payloads
            }
            .matches(1)
        );
        // A point with no payload at all also passes a pure must_not filter.
        assert!(
            FilterQuery {
                filter: &f,
                payloads: &payloads
            }
            .matches(99)
        );

        // must + must_not + should evaluated together on one payload, all satisfied.
        let combined = Filter {
            must: vec![Condition {
                key: "tag".into(),
                r#match: Some(json!("b")),
                range: None,
            }],
            should: vec![Condition {
                key: "tag".into(),
                r#match: Some(json!("b")),
                range: None,
            }],
            must_not: vec![Condition {
                key: "missing".into(),
                r#match: Some(json!("z")),
                range: None,
            }],
        };
        assert!(
            FilterQuery {
                filter: &combined,
                payloads: &payloads
            }
            .matches(1)
        );
    }
}
