use serde::{Deserialize, Serialize};

/// Types attached to operation signatures. Used by the type-checker to
/// verify well-formed compilation chains.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Type {
    /// Primitive string.
    Str,
    /// Primitive numeric (f64).
    Num,
    /// Primitive boolean.
    Bool,
    /// Unit / empty type.
    Unit,
    /// Homogeneous list.
    List(Box<Type>),
    /// Named record type provided by a domain, e.g., "ProteinRecord".
    Named(String),
    /// Type variable for parametric polymorphism (future use).
    Var(String),
}

impl Type {
    pub fn named<S: Into<String>>(s: S) -> Self {
        Type::Named(s.into())
    }

    pub fn list(inner: Type) -> Self {
        Type::List(Box::new(inner))
    }
}
