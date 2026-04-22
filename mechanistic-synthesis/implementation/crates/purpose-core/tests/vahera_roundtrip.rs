use std::collections::BTreeMap;

use purpose_core::{typecheck::typecheck, Operation, Type, VaHera, Value};

fn sample_fragment() -> VaHera {
    let mut lookup_args = BTreeMap::new();
    lookup_args.insert("gene".into(), VaHera::Literal(Value::Str("SOD1".into())));
    VaHera::Compose(vec![
        VaHera::Call {
            op: "lookup_protein_by_gene".into(),
            args: lookup_args,
        },
        VaHera::Call {
            op: "summarize_protein".into(),
            args: BTreeMap::new(),
        },
    ])
}

fn sample_ops() -> std::collections::HashMap<String, Operation> {
    let mut lookup_inputs = BTreeMap::new();
    lookup_inputs.insert("gene".into(), Type::Str);
    lookup_inputs.insert("organism".into(), Type::Str);
    let mut summary_inputs = BTreeMap::new();
    summary_inputs.insert("input".into(), Type::named("ProteinRecord"));

    let ops = vec![
        Operation::new(
            "lookup_protein_by_gene",
            lookup_inputs,
            Type::named("ProteinRecord"),
            "look up protein",
        ),
        Operation::new(
            "summarize_protein",
            summary_inputs,
            Type::Str,
            "summarise record",
        ),
    ];
    ops.into_iter().map(|op| (op.name.clone(), op)).collect()
}

#[test]
fn fragment_serialises_and_roundtrips() {
    let f = sample_fragment();
    let json = serde_json::to_string(&f).expect("serialise");
    let back: VaHera = serde_json::from_str(&json).expect("deserialise");
    assert_eq!(f, back);
}

#[test]
fn fragment_typechecks() {
    let f = sample_fragment();
    let ops = sample_ops();
    let ty = typecheck(&f, &ops).expect("typecheck");
    assert_eq!(ty, Type::Str);
}

#[test]
fn fragment_with_hole_fails_typecheck() {
    let mut args = BTreeMap::new();
    args.insert("gene".into(), VaHera::Hole("gene".into()));
    let f = VaHera::Call {
        op: "lookup_protein_by_gene".into(),
        args,
    };
    assert!(typecheck(&f, &sample_ops()).is_err());
}

#[test]
fn unresolved_hole_detected() {
    let f = VaHera::Hole("pending".into());
    assert!(!f.is_fully_resolved());

    let complete = VaHera::Literal(Value::Str("done".into()));
    assert!(complete.is_fully_resolved());
}
