use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;

use purpose_core::{Error, VaHera, Value};
use tracing::debug;

use crate::registry::OperationRegistry;

/// Walks a vaHera fragment and dispatches each call to its registered
/// provider. Composition threads each step's output into the next step's
/// `input` slot unless already bound.
pub struct Executor {
    registry: OperationRegistry,
}

impl Executor {
    pub fn new(registry: OperationRegistry) -> Self {
        Self { registry }
    }

    pub fn registry(&self) -> &OperationRegistry {
        &self.registry
    }

    /// Execute a fragment with no piped input.
    pub async fn execute(&self, program: &VaHera) -> Result<Value, Error> {
        self.execute_with(program, None).await
    }

    /// Execute a fragment, with `piped` available to fill the next call's
    /// `input` slot if that slot is otherwise absent.
    pub fn execute_with<'a>(
        &'a self,
        program: &'a VaHera,
        piped: Option<Value>,
    ) -> Pin<Box<dyn Future<Output = Result<Value, Error>> + Send + 'a>> {
        Box::pin(async move {
            match program {
                VaHera::Hole(name) => Err(Error::Internal(format!(
                    "unresolved hole at execution time: {}",
                    name
                ))),

                VaHera::Literal(v) => Ok(v.clone()),

                VaHera::Call { op, args } => {
                    let mut resolved: BTreeMap<String, Value> = BTreeMap::new();
                    for (k, v) in args {
                        let val = self.execute_with(v, None).await?;
                        resolved.insert(k.clone(), val);
                    }
                    if let Some(p) = piped {
                        resolved
                            .entry("input".to_string())
                            .or_insert(p);
                    }
                    let (_, provider) = self
                        .registry
                        .get(op)
                        .ok_or_else(|| Error::Internal(format!(
                            "unregistered operation: '{}'",
                            op
                        )))?;
                    debug!(op = %op, "dispatching");
                    provider.invoke(op, &resolved).await
                }

                VaHera::Compose(steps) => {
                    let mut current: Option<Value> = None;
                    for step in steps {
                        let next = self
                            .execute_with(step, current.clone())
                            .await?;
                        current = Some(next);
                    }
                    Ok(current.unwrap_or(Value::Null))
                }
            }
        })
    }
}
