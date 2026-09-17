use crate::contract::Verifier;
use crate::source::Document;
use crate::tokenizer::Tokenizer;

/// One training example: token ids for the full sequence (prompt followed by
/// completion). The training loop predicts each token from everything
/// before it, so no separate prompt/completion split is needed at the
/// tensor level — the split matters only for the verifier.
pub struct Example {
    pub token_ids: Vec<u32>,
}

pub struct Corpus {
    pub tokenizer: Tokenizer,
    pub examples: Vec<Example>,
}

/// Splits ingested documents into windows and keeps only the ones the
/// verifier admits — the architecture-agnostic half of corpus building.
/// Shared by both the from-scratch and pretrained training paths, which
/// differ only in how the admitted text is tokenized afterward.
pub fn admitted_texts(docs: &[Document], verifier: &dyn Verifier, window_chars: usize) -> Vec<String> {
    let mut admitted = Vec::new();
    for doc in docs {
        for window in windows(&doc.text, window_chars) {
            let mid = window.len() / 2;
            let (prompt, completion) = window.split_at(mid.max(1).min(window.len()));
            if verifier.verify(prompt, completion) {
                admitted.push(window.to_string());
            }
        }
    }
    admitted
}

/// Builds a from-scratch training corpus: windows and verifies documents,
/// then trains a fresh word-level `Tokenizer` on exactly the admitted text
/// (so the vocabulary is scoped to this theme) and tokenizes with it.
pub fn build(
    docs: &[Document],
    verifier: &dyn Verifier,
    vocab_size: usize,
    block_size: usize,
) -> Corpus {
    let admitted = admitted_texts(docs, verifier, block_size * 4);
    let tokenizer = Tokenizer::train(&admitted, vocab_size);

    let examples = admitted
        .iter()
        .map(|text| {
            let mut ids = tokenizer.encode(text);
            ids.push(tokenizer.eos_id);
            ids.truncate(block_size);
            Example { token_ids: ids }
        })
        .filter(|ex| ex.token_ids.len() >= 8)
        .collect();

    Corpus {
        tokenizer,
        examples,
    }
}

/// Splits `text` into roughly `max_chars`-sized windows on paragraph
/// boundaries where possible, falling back to a hard cut.
fn windows(text: &str, max_chars: usize) -> Vec<&str> {
    if text.len() <= max_chars {
        return vec![text];
    }

    let mut out = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    while start < bytes.len() {
        let end = (start + max_chars).min(bytes.len());
        // Walk back to a char boundary and, if possible, a paragraph break.
        let mut cut = end;
        while cut > start && !text.is_char_boundary(cut) {
            cut -= 1;
        }
        if let Some(rel) = text[start..cut].rfind("\n\n") {
            if start + rel > start {
                cut = start + rel;
            }
        }
        if cut <= start {
            cut = end;
        }
        out.push(&text[start..cut]);
        start = cut;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::HeuristicVerifier;
    use std::collections::BTreeMap;

    #[test]
    fn builds_examples_from_documents() {
        let docs = vec![Document {
            id: "1".into(),
            text: "The extremal regime exercises every degree of freedom of the domain's action space under a single scalar objective, which is the central claim of the inclusion theorem.".into(),
            metadata: BTreeMap::new(),
        }];
        let verifier = HeuristicVerifier::default();
        let corpus = build(&docs, &verifier, 512, 128);
        assert!(!corpus.examples.is_empty());
        assert!(corpus.tokenizer.vocab_size() > 256);
    }

    #[test]
    fn rejects_degenerate_windows() {
        let docs = vec![Document {
            id: "1".into(),
            text: "no no no no no no no no no no no no no no no no".into(),
            metadata: BTreeMap::new(),
        }];
        let verifier = HeuristicVerifier::default();
        let corpus = build(&docs, &verifier, 512, 128);
        assert!(corpus.examples.is_empty());
    }
}
