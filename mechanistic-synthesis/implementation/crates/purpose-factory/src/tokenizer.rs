use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A minimal whitespace/punctuation word-piece tokenizer built from corpus
/// frequency statistics, with single-byte fallback for anything outside the
/// learned vocabulary. Deliberately simple (no BPE merges) to keep the
/// factory dependency-free of an external tokenizer training pipeline;
/// sufficient for a from-scratch small model trained on one theme's corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tokenizer {
    token_to_id: HashMap<String, u32>,
    id_to_token: Vec<String>,
    pub eos_id: u32,
}

const EOS_TOKEN: &str = "<|eos|>";
const UNK_TOKEN: &str = "<|unk|>";

impl Tokenizer {
    /// Builds a vocabulary from the most frequent word-level tokens in the
    /// corpus, capped at `vocab_size` (including special tokens and the
    /// 256-entry byte fallback range).
    pub fn train(texts: &[String], vocab_size: usize) -> Self {
        let mut freq: HashMap<String, usize> = HashMap::new();
        for text in texts {
            for word in split_words(text) {
                *freq.entry(word).or_insert(0) += 1;
            }
        }

        let mut ranked: Vec<(String, usize)> = freq.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let mut id_to_token = vec![EOS_TOKEN.to_string(), UNK_TOKEN.to_string()];
        // Reserve byte fallback range so any raw text is representable even
        // when a word wasn't seen at training time.
        for b in 0u16..256 {
            id_to_token.push(format!("<byte:{b:02x}>"));
        }
        let reserved = id_to_token.len();
        let budget = vocab_size.saturating_sub(reserved);

        for (word, _count) in ranked.into_iter().take(budget) {
            id_to_token.push(word);
        }

        let token_to_id = id_to_token
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();

        Self {
            token_to_id,
            id_to_token,
            eos_id: 0,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for word in split_words(text) {
            if let Some(&id) = self.token_to_id.get(&word) {
                ids.push(id);
            } else {
                for byte in word.as_bytes() {
                    let key = format!("<byte:{byte:02x}>");
                    ids.push(*self.token_to_id.get(&key).unwrap());
                }
            }
        }
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            if id == self.eos_id {
                continue;
            }
            if let Some(tok) = self.id_to_token.get(id as usize) {
                if let Some(hex) = tok.strip_prefix("<byte:").and_then(|s| s.strip_suffix('>')) {
                    if let Ok(byte) = u8::from_str_radix(hex, 16) {
                        // Best-effort: push raw byte; may produce a mojibake
                        // boundary for multi-byte UTF-8 split across tokens,
                        // acceptable for this minimal v1 tokenizer.
                        out.push(byte as char);
                        continue;
                    }
                }
                out.push_str(tok);
                out.push(' ');
            }
        }
        out.trim().to_string()
    }
}

fn split_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !current.is_empty() {
                words.push(std::mem::take(&mut current));
            }
        } else if ch.is_alphanumeric() || ch == '\'' {
            current.push(ch.to_ascii_lowercase());
        } else {
            if !current.is_empty() {
                words.push(std::mem::take(&mut current));
            }
            words.push(ch.to_string());
        }
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_known_words() {
        let corpus = vec!["the quick brown fox jumps over the lazy dog".to_string()];
        let tok = Tokenizer::train(&corpus, 300);
        let ids = tok.encode("the quick fox");
        assert!(!ids.is_empty());
        let decoded = tok.decode(&ids);
        assert!(decoded.contains("quick"));
    }

    #[test]
    fn falls_back_to_bytes_for_unseen_words() {
        let corpus = vec!["alpha beta gamma".to_string()];
        let tok = Tokenizer::train(&corpus, 300);
        let ids = tok.encode("zzzznotseen");
        assert!(!ids.is_empty());
    }
}
