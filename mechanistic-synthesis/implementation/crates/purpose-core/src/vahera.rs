use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The vaHera abstract syntax tree.
///
/// A vaHera expression is either a typed operation call, a sequential
/// composition of expressions, a carried literal value, or an unresolved
/// hole. A fully-resolved fragment contains no holes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum VaHera {
    /// A typed operation call with named arguments.
    Call {
        op: String,
        #[serde(default)]
        args: BTreeMap<String, VaHera>,
    },

    /// Sequential composition. The output of each step is threaded into
    /// the conventional `input` slot of the next step unless that slot is
    /// already bound.
    Compose(Vec<VaHera>),

    /// A carried literal value (string, number, record, etc.).
    Literal(Value),

    /// An unresolved subproblem. Execution rejects fragments containing holes.
    Hole(String),
}

impl VaHera {
    /// Convenience constructor: a call with no arguments.
    pub fn call<N: Into<String>>(name: N) -> Self {
        VaHera::Call { op: name.into(), args: BTreeMap::new() }
    }

    /// Returns true if this fragment (and all sub-expressions) contain no
    /// unresolved holes.
    pub fn is_fully_resolved(&self) -> bool {
        match self {
            VaHera::Hole(_) => false,
            VaHera::Literal(_) => true,
            VaHera::Call { args, .. } => args.values().all(|v| v.is_fully_resolved()),
            VaHera::Compose(steps) => steps.iter().all(|s| s.is_fully_resolved()),
        }
    }
}

/// A runtime value carried through vaHera execution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    List(Vec<Value>),
    Record(BTreeMap<String, Value>),
}

impl Value {
    pub fn str<S: Into<String>>(s: S) -> Self {
        Value::Str(s.into())
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_record(&self) -> Option<&BTreeMap<String, Value>> {
        match self {
            Value::Record(r) => Some(r),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&[Value]> {
        match self {
            Value::List(l) => Some(l.as_slice()),
            _ => None,
        }
    }
}
