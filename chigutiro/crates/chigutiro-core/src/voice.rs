//! The voice corpus: the only path by which anything reaches model weights.
//!
//! Admission is by kind and authorship, not by content: only prose the owner
//! wrote is eligible. Measurements, money, positions, contacts and other
//! people's writing are recall-only — weights cannot be queried for a
//! current value, cannot be kept current, and cannot forget one person. What
//! is admitted is cleaned (quoted replies and signatures cut) and redacted
//! (addresses, account and card numbers, codes, credentials) first.

use std::sync::OnceLock;

use regex::{Captures, Regex};
use serde::Serialize;

use crate::record::{Body, Stored};

/// Below this, a cleaned message is a pleasantry, not a sample of voice.
const MIN_VOICE_CHARS: usize = 40;

#[derive(Debug, Clone, Serialize)]
pub struct VoiceDoc {
    pub id: String,
    pub seq: u64,
    pub text: String,
}

pub fn voice_doc(stored: &Stored) -> Option<VoiceDoc> {
    let Body::Prose { text, title, authored_by_owner: true } = &stored.record.body else { return None };
    let body = clean(text);
    let text = match title.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
        Some(t) => format!("{t}\n\n{body}"),
        None => body,
    };
    let text = redact(&text);
    (text.trim().chars().count() >= MIN_VOICE_CHARS).then(|| VoiceDoc { id: stored.id.clone(), seq: stored.seq, text })
}

/// Cuts what the owner did not write in this message: quoted history
/// (`> ...`, "On ... wrote:", "Am ... schrieb ...:", forwarded headers) and
/// everything after a signature delimiter.
pub fn clean(text: &str) -> String {
    static HISTORY: OnceLock<Regex> = OnceLock::new();
    let history = HISTORY.get_or_init(|| {
        Regex::new(
            r"(?im)^\s*(on\s.{0,200}\swrote:|am\s.{0,200}\sschrieb.{0,120}:|-{2,}\s*(original message|forwarded message|ursprüngliche nachricht|weitergeleitete nachricht)\s*-{2,}|from:\s.+\n\s*sent:)",
        )
        .expect("history regex")
    });
    let text = text.replace("\r\n", "\n");
    let text = match history.find(&text) {
        Some(m) => &text[..m.start()],
        None => &text[..],
    };
    let mut kept = Vec::new();
    for line in text.lines() {
        // RFC 3676 signature delimiter, and the common bare variant.
        if line == "-- " || line.trim() == "--" {
            break;
        }
        if line.trim_start().starts_with('>') {
            continue;
        }
        kept.push(line);
    }
    kept.join("\n").trim().to_string()
}

struct Rules {
    email: Regex,
    url_secret: Regex,
    iban: Regex,
    card: Regex,
    phone: Regex,
    code: Regex,
    credential: Regex,
}

fn rules() -> &'static Rules {
    static RULES: OnceLock<Rules> = OnceLock::new();
    RULES.get_or_init(|| Rules {
        email: Regex::new(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}").unwrap(),
        url_secret: Regex::new(
            r"(?i)https?://\S*[?&](token|key|sig|signature|auth|code|access_token|session|password)=\S*",
        )
        .unwrap(),
        iban: Regex::new(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?\b").unwrap(),
        card: Regex::new(r"\b\d(?:[ \-]?\d){12,18}\b").unwrap(),
        phone: Regex::new(r"(?:\+|\b00)\d[\d \-/()]{7,}\d\b|\b0\d{2,5}[ /\-]?\d{3,}(?:[ \-]?\d+)*\b").unwrap(),
        code: Regex::new(
            r"(?i)\b(code|pin|tan|otp|passcode|verification|bestätigungscode|sicherheitscode)\b([^0-9\n]{0,24})\d{4,8}\b",
        )
        .unwrap(),
        credential: Regex::new(
            r"(?im)\b(password|passwort|kennwort|pwd|api[_ ]?key|secret|token)\b(\s*[:=]\s*)\S+",
        )
        .unwrap(),
    })
}

pub fn redact(text: &str) -> String {
    let r = rules();
    let t = r.url_secret.replace_all(text, "<url>");
    let t = r.email.replace_all(&t, "<email>");
    let t = r.credential.replace_all(&t, |c: &Captures| format!("{}{}<redacted>", &c[1], &c[2]));
    let t = r.code.replace_all(&t, |c: &Captures| format!("{}{}<code>", &c[1], &c[2]));
    let t = r.iban.replace_all(&t, "<iban>");
    let t = r.card.replace_all(&t, |c: &Captures| {
        let digits: Vec<u32> = c[0].chars().filter_map(|ch| ch.to_digit(10)).collect();
        if luhn(&digits) { "<card>".to_string() } else { c[0].to_string() }
    });
    let t = r.phone.replace_all(&t, "<phone>");
    t.into_owned()
}

fn luhn(digits: &[u32]) -> bool {
    let sum: u32 = digits
        .iter()
        .rev()
        .enumerate()
        .map(|(i, &d)| if i % 2 == 1 { let x = d * 2; if x > 9 { x - 9 } else { x } } else { d })
        .sum();
    sum.is_multiple_of(10)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuts_quoted_history_and_signature() {
        let mail = "Thanks, I'll send the draft on Monday.\n\n-- \nKundai\n\nOn Tue, 3 Sep 2026, Mark wrote:\n> can you send it?";
        assert_eq!(clean(mail), "Thanks, I'll send the draft on Monday.");
        let de = "Passt, bis morgen.\n\nAm 03.09.2026 um 10:00 schrieb Mark Dörr <m@x.de>:\n> Treffen?";
        assert_eq!(clean(de), "Passt, bis morgen.");
    }

    #[test]
    fn redacts_identifiers_and_secrets() {
        let t = redact(
            "Write to alice@example.org, IBAN DE89 3704 0044 0532 0130 00, card 4111 1111 1111 1111, \
             call +49 30 1234 5678. Your TAN: 482913. password: hunter2. \
             See https://x.io/reset?token=abc123 — order 1234567890123 stays.",
        );
        for leaked in ["alice@", "DE89", "4111", "1234 5678", "482913", "hunter2", "abc123"] {
            assert!(!t.contains(leaked), "leaked {leaked}: {t}");
        }
        // A 13-digit number that fails Luhn is not a card and is kept.
        assert!(t.contains("1234567890123"), "{t}");
        assert!(t.contains("TAN: <code>") && t.contains("password: <redacted>"), "{t}");
    }
}
