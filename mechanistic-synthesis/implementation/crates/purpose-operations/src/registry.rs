use std::collections::HashMap;
use std::sync::Arc;

use purpose_core::Operation;

use crate::provider::Provider;

/// Maps operation names to their signature and provider.
///
/// Cloning is cheap (providers are behind `Arc`). A registry is built once
/// at startup and shared read-only across concurrent query execution.
#[derive(Clone)]
pub struct OperationRegistry {
    entries: HashMap<String, (Operation, Arc<dyn Provider>)>,
}

impl OperationRegistry {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn register(&mut self, op: Operation, provider: Arc<dyn Provider>) {
        self.entries.insert(op.name.clone(), (op, provider));
    }

    pub fn get(&self, name: &str) -> Option<(&Operation, &Arc<dyn Provider>)> {
        self.entries.get(name).map(|(op, p)| (op, p))
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(|s| s.as_str())
    }

    pub fn operations(&self) -> impl Iterator<Item = &Operation> {
        self.entries.values().map(|(op, _)| op)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for OperationRegistry {
    fn default() -> Self {
        Self::new()
    }
}
