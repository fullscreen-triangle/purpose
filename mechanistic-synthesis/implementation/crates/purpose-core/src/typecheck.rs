use std::collections::HashMap;

use crate::error::Error;
use crate::operation::Operation;
use crate::types::Type;
use crate::vahera::{VaHera, Value};

/// A minimal type-checker for vaHera fragments.
///
/// Given a registry of known operations, verifies that every call
/// references a known operation, every literal argument has the expected
/// type, and every composition threads the output of one step into the
/// `input` slot of the next step compatibly.
pub fn typecheck(
    fragment: &VaHera,
    ops: &HashMap<String, Operation>,
) -> Result<Type, Error> {
    match fragment {
        VaHera::Hole(name) => Err(Error::Type(format!(
            "hole '{}' remains in fragment: resolution incomplete",
            name
        ))),
        VaHera::Literal(v) => Ok(literal_type(v)),
        VaHera::Call { op, args } => {
            let signature = ops.get(op).ok_or_else(|| {
                Error::Type(format!("call to unknown operation '{}'", op))
            })?;
            for (arg_name, arg_expr) in args {
                let expected = signature.inputs.get(arg_name).ok_or_else(|| {
                    Error::Type(format!(
                        "operation '{}' does not accept argument '{}'",
                        op, arg_name
                    ))
                })?;
                let actual = typecheck(arg_expr, ops)?;
                if !types_compatible(expected, &actual) {
                    return Err(Error::Type(format!(
                        "argument '{}' of '{}': expected {:?}, got {:?}",
                        arg_name, op, expected, actual
                    )));
                }
            }
            Ok(signature.output.clone())
        }
        VaHera::Compose(steps) => {
            let mut last = Type::Unit;
            for (i, step) in steps.iter().enumerate() {
                // Each step's `input` (if declared in its signature) should
                // accept the previous step's output type.
                if i > 0 {
                    if let VaHera::Call { op, .. } = step {
                        if let Some(sig) = ops.get(op) {
                            if let Some(expected_input) = sig.inputs.get("input") {
                                if !types_compatible(expected_input, &last) {
                                    return Err(Error::Type(format!(
                                        "compose: step {} '{}' expects input {:?}, got {:?}",
                                        i, op, expected_input, last
                                    )));
                                }
                            }
                        }
                    }
                }
                last = typecheck(step, ops)?;
            }
            Ok(last)
        }
    }
}

fn literal_type(v: &Value) -> Type {
    match v {
        Value::Null => Type::Unit,
        Value::Bool(_) => Type::Bool,
        Value::Num(_) => Type::Num,
        Value::Str(_) => Type::Str,
        Value::List(_) => Type::list(Type::Named("Unknown".into())),
        Value::Record(_) => Type::Named("Unknown".into()),
    }
}

/// A permissive compatibility check. Exact match passes; a Named("Unknown")
/// matches any Named type (for literals produced by the parser without full
/// context).
fn types_compatible(expected: &Type, actual: &Type) -> bool {
    if expected == actual {
        return true;
    }
    matches!(
        (expected, actual),
        (Type::Named(_), Type::Named(n)) if n == "Unknown"
    ) || matches!(
        (expected, actual),
        (Type::Named(n), Type::Named(_)) if n == "Unknown"
    )
}
