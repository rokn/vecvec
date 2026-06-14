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
}
