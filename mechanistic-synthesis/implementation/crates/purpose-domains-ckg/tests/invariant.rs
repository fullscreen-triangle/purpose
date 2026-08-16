// binv:invariant against the real repository graph.
// Run with: cargo test -p purpose-domains-ckg --test invariant -- --nocapture
use std::collections::BTreeMap;
use std::path::Path;

#[test]
fn the_character_is_conserved_under_relabelling_the_real_repository() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let stored = match purpose_domains_ckg::load(root) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("no .purpose/ckg.json — run `purpose ckg build` first; skipping");
            return;
        }
    };
    let g = stored.graph().expect("stored graph must load");
    let items = g.items();
    assert!(items.len() > 10, "expected a real repository graph");

    // Permute every module name; σ is structural, so its multiset must not move.
    let perm: BTreeMap<String, String> = items
        .iter()
        .enumerate()
        .map(|(i, v)| (v.to_string(), format!("module-{i:04}")))
        .collect();
    let h = g.relabel(&perm).expect("relabelling must succeed");

    let chi_g = purpose_ckg::character(&g);
    let chi_h = purpose_ckg::character(&h);
    assert_eq!(chi_g, chi_h, "χ moved under a renaming of {} modules", items.len());
    assert_ne!(g.items(), h.items(), "the labels genuinely moved");
    eprintln!(
        "χ conserved over {} modules; floor {:?}",
        items.len(),
        purpose_ckg::system_floor(&g)
    );
}
