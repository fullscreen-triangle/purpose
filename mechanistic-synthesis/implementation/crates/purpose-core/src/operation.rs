use crate::types::Type;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A typed operation primitive available to a domain.
///
/// Operations are the leaves of vaHera compilation. A `Resolver` emits a
/// fragment whose calls name operations from its `Domain::operations` set;
/// the runtime dispatches each call to a registered provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    pub name: String,
    pub inputs: BTreeMap<String, Type>,
    pub output: Type,
    pub description: String,
}

impl Operation {
    pub fn new<N: Into<String>, D: Into<String>>(
        name: N,
        inputs: BTreeMap<String, Type>,
        output: Type,
        description: D,
    ) -> Self {
        Self {
            name: name.into(),
            inputs,
            output,
            description: description.into(),
        }
    }
}
