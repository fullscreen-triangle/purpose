//! One tokenizer for every receiver, so a query term means the same thing to
//! the lexical index as to the metric, account, and person matchers.

use std::collections::BTreeSet;

const STOPWORDS: &[&str] = &[
    // English
    "the", "and", "for", "are", "was", "were", "has", "have", "had", "this", "that", "with", "from",
    "what", "when", "how", "did", "does", "can", "could", "would", "should", "about", "into", "than",
    "then", "them", "they", "their", "there", "been", "being", "will", "shall", "not", "but", "you",
    "your", "our", "its", "his", "her", "him", "who", "whom", "which", "while", "any", "all",
    "my", "me", "is", "it", "of", "to", "in", "on", "at", "by", "an", "or", "as", "be", "do", "so",
    "if", "no", "up", "we", "am", "i", "a",
    // German
    "der", "die", "das", "und", "ist", "ich", "mit", "für", "den", "dem", "des", "ein", "eine",
    "einen", "nicht", "auf", "sie", "wie", "was", "wir", "auch", "sich", "noch", "nach", "bei",
];

pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .filter(|t| t.chars().count() >= 2 && !STOPWORDS.contains(&t.as_str()))
        .collect()
}

/// Distinct query terms — the unit every receiver's coverage is measured in.
pub fn query_terms(text: &str) -> BTreeSet<String> {
    tokenize(text).into_iter().collect()
}

/// Fraction of `query` terms present in `candidate`, in (0, 1].
pub fn coverage(query: &BTreeSet<String>, candidate: &BTreeSet<String>) -> f64 {
    if query.is_empty() {
        return 0.0;
    }
    let hit = query.iter().filter(|t| candidate.contains(*t)).count();
    hit as f64 / query.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_punctuation_and_drops_stopwords() {
        assert_eq!(tokenize("What's my HRV, über-trend?"), vec!["hrv", "über", "trend"]);
    }
}
