pub mod calendar;
pub mod ledger;
pub mod people;
pub mod places;
pub mod series;
pub mod text;

use chrono::{DateTime, Utc};

/// Precision that reads naturally at the value's own scale.
pub(crate) fn num(v: f64) -> String {
    let a = v.abs();
    if a >= 100.0 {
        format!("{v:.0}")
    } else if a >= 10.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    }
}

pub(crate) fn money(v: f64, currency: &str) -> String {
    format!("{v:+.2} {currency}")
}

pub(crate) fn day(ts: DateTime<Utc>) -> String {
    ts.format("%Y-%m-%d").to_string()
}

/// "today", "3 days ago", "in 12 days" — so a stale reading is never
/// presented as if it were current.
pub(crate) fn relative(ts: DateTime<Utc>, now: DateTime<Utc>) -> String {
    let days = (now.date_naive() - ts.date_naive()).num_days();
    match days {
        0 => "today".into(),
        1 => "yesterday".into(),
        -1 => "tomorrow".into(),
        d if d > 0 => format!("{d} days ago"),
        d => format!("in {} days", -d),
    }
}
