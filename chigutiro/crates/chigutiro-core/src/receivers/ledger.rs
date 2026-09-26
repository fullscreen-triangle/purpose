//! Financial receiver: account statements (cash, investments, debt) and
//! transactions. Every figure is computed from the records at query time and
//! is labelled with how it was derived; currencies are never summed together.

use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Duration, Utc};

use super::{day, money, relative};
use crate::claim::Candidate;
use crate::record::{Body, Stored};
use crate::terms::{coverage, tokenize};

const VOCAB: &[&str] = &[
    "money", "balance", "balances", "bank", "account", "accounts", "spend", "spent", "spending", "cost",
    "costs", "paid", "pay", "income", "salary", "debt", "debts", "loan", "loans", "invest", "investment",
    "investments", "portfolio", "savings", "save", "afford", "budget", "finance", "finances", "net",
    "worth", "cash", "geld", "konto", "kontostand", "ausgaben", "schulden", "gehalt",
];

struct Tx {
    ts: DateTime<Utc>,
    amount: f64,
    currency: String,
    label: String,
    tokens: BTreeSet<String>,
    source: String,
    record_id: String,
}

struct Statement {
    ts: DateTime<Utc>,
    amount: f64,
    currency: String,
    class: Option<String>,
    source: String,
    record_id: String,
}

#[derive(Default)]
struct Account {
    txs: Vec<Tx>,
    statements: Vec<Statement>,
}

#[derive(Default)]
pub struct Ledger {
    accounts: BTreeMap<String, Account>,
}

impl Ledger {
    pub fn add(&mut self, stored: &Stored) {
        let r = &stored.record;
        match &r.body {
            Body::Transaction { account, amount, currency, counterparty, memo, category } => {
                let label = [counterparty, category, memo]
                    .iter()
                    .filter_map(|s| s.as_deref())
                    .find(|s| !s.trim().is_empty())
                    .unwrap_or("(unlabelled)")
                    .to_string();
                let text = [counterparty, memo, category].iter().filter_map(|s| s.as_deref()).collect::<Vec<_>>().join(" ");
                self.accounts.entry(account.clone()).or_default().txs.push(Tx {
                    ts: r.ts,
                    amount: *amount,
                    currency: currency.clone(),
                    label,
                    tokens: tokenize(&text).into_iter().collect(),
                    source: r.source.clone(),
                    record_id: stored.id.clone(),
                });
            }
            Body::Balance { account, amount, currency, class } => {
                self.accounts.entry(account.clone()).or_default().statements.push(Statement {
                    ts: r.ts,
                    amount: *amount,
                    currency: currency.clone(),
                    class: class.clone(),
                    source: r.source.clone(),
                    record_id: stored.id.clone(),
                });
            }
            _ => {}
        }
    }

    pub fn account_count(&self) -> usize {
        self.accounts.len()
    }

    pub fn candidates(&self, query: &BTreeSet<String>, now: DateTime<Utc>) -> Vec<Candidate> {
        let vocab: BTreeSet<String> = VOCAB.iter().map(|s| s.to_string()).collect();
        let mut out = Vec::new();
        let mut net: BTreeMap<(String, String), f64> = BTreeMap::new();
        let mut net_sources = BTreeSet::new();
        let mut net_ids = Vec::new();

        for (name, acct) in &self.accounts {
            let mut acct_vocab = vocab.clone();
            acct_vocab.extend(tokenize(name));
            let latest = acct.statements.iter().filter(|s| s.ts <= now).max_by_key(|s| s.ts);
            if let Some(cls) = latest.and_then(|s| s.class.as_deref()) {
                acct_vocab.extend(tokenize(cls));
            }
            let cov = coverage(query, &acct_vocab);

            if let Some(st) = latest {
                let since: Vec<&Tx> = acct.txs.iter().filter(|t| t.ts > st.ts && t.ts <= now && t.currency == st.currency).collect();
                let moved: f64 = since.iter().map(|t| t.amount).sum();
                let class = st.class.as_deref().map(|c| format!(" ({c})")).unwrap_or_default();
                let mut text = format!(
                    "account {name}{class}: statement {} on {} ({}) [{}]",
                    money(st.amount, &st.currency),
                    day(st.ts),
                    relative(st.ts, now),
                    st.source
                );
                if !since.is_empty() {
                    text.push_str(&format!(
                        "; {} transactions booked since, net {}, so statement + since = {}",
                        since.len(),
                        money(moved, &st.currency),
                        money(st.amount + moved, &st.currency)
                    ));
                }
                *net.entry((st.currency.clone(), st.class.clone().unwrap_or_else(|| "unclassified".into()))).or_default() +=
                    st.amount + moved;
                net_sources.insert(st.source.clone());
                net_ids.push(st.record_id.clone());
                if cov > 0.0 {
                    let mut sources = BTreeSet::from([st.source.clone()]);
                    sources.extend(since.iter().map(|t| t.source.clone()));
                    let mut ids = vec![st.record_id.clone()];
                    ids.extend(since.iter().map(|t| t.record_id.clone()));
                    out.push(Candidate {
                        receiver: "ledger",
                        text,
                        coverage: cov,
                        rank_score: 2.0,
                        sources,
                        record_ids: ids,
                        contested: None,
                    });
                }
            }

            let recent: Vec<&Tx> = acct.txs.iter().filter(|t| t.ts > now - Duration::days(30) && t.ts <= now).collect();
            if cov > 0.0 && !recent.is_empty() {
                out.push(flow_candidate(name, &recent, cov));
            }
        }

        if !net.is_empty() {
            let mut totals_vocab = vocab.clone();
            totals_vocab.extend(["total", "overall", "everything", "all"].map(String::from));
            let cov = coverage(query, &totals_vocab);
            if cov > 0.0 {
                let parts: Vec<String> = net.iter().map(|((cur, cls), v)| format!("{cls} {}", money(*v, cur))).collect();
                out.push(Candidate {
                    receiver: "ledger",
                    text: format!(
                        "position across {} account(s), latest statement plus later bookings: {}",
                        self.accounts.values().filter(|a| !a.statements.is_empty()).count(),
                        parts.join(", ")
                    ),
                    coverage: cov,
                    rank_score: 3.0,
                    sources: net_sources,
                    record_ids: net_ids,
                    contested: None,
                });
            }
        }

        out.extend(self.matching_payees(query, now));
        out
    }

    /// Terms that hit a counterparty, memo or category: "how much at REWE".
    fn matching_payees(&self, query: &BTreeSet<String>, now: DateTime<Utc>) -> Vec<Candidate> {
        let mut out = Vec::new();
        for term in query {
            if VOCAB.contains(&term.as_str()) {
                continue;
            }
            let hits: Vec<&Tx> = self
                .accounts
                .values()
                .flat_map(|a| a.txs.iter())
                .filter(|t| t.ts > now - Duration::days(90) && t.ts <= now && t.tokens.contains(term))
                .collect();
            if hits.is_empty() {
                continue;
            }
            let mut by_cur: BTreeMap<&str, f64> = BTreeMap::new();
            for t in &hits {
                *by_cur.entry(&t.currency).or_default() += t.amount;
            }
            let last = hits.iter().max_by_key(|t| t.ts).expect("non-empty");
            let totals: Vec<String> = by_cur.iter().map(|(c, v)| money(*v, c)).collect();
            let mut vocab: BTreeSet<String> = VOCAB.iter().map(|s| s.to_string()).collect();
            vocab.insert(term.clone());
            out.push(Candidate {
                receiver: "ledger",
                text: format!(
                    "transactions matching '{term}' in the last 90 days: {} totalling {}; last {} on {}",
                    hits.len(),
                    totals.join(", "),
                    last.label,
                    day(last.ts)
                ),
                coverage: coverage(query, &vocab),
                rank_score: hits.len() as f64,
                sources: hits.iter().map(|t| t.source.clone()).collect(),
                record_ids: hits.iter().map(|t| t.record_id.clone()).collect(),
                contested: None,
            });
        }
        out
    }
}

fn flow_candidate(name: &str, recent: &[&Tx], cov: f64) -> Candidate {
    let mut by_cur: BTreeMap<&str, (f64, f64)> = BTreeMap::new();
    for t in recent {
        let e = by_cur.entry(&t.currency).or_default();
        if t.amount >= 0.0 {
            e.0 += t.amount;
        } else {
            e.1 += t.amount;
        }
    }
    let mut out_by_label: BTreeMap<(&str, &str), f64> = BTreeMap::new();
    for t in recent.iter().filter(|t| t.amount < 0.0) {
        *out_by_label.entry((&t.label, &t.currency)).or_default() += t.amount;
    }
    let mut largest: Vec<_> = out_by_label.into_iter().collect();
    largest.sort_by(|a, b| a.1.total_cmp(&b.1));
    let flows: Vec<String> = by_cur
        .iter()
        .map(|(c, (i, o))| format!("in {}, out {}, net {}", money(*i, c), money(*o, c), money(i + o, c)))
        .collect();
    let top: Vec<String> = largest.iter().take(3).map(|((l, c), v)| format!("{l} {}", money(*v, c))).collect();
    let mut text = format!("account {name}, last 30 days ({} transactions): {}", recent.len(), flows.join("; "));
    if !top.is_empty() {
        text.push_str(&format!("; largest outflows: {}", top.join(", ")));
    }
    Candidate {
        receiver: "ledger",
        text,
        coverage: cov,
        rank_score: 1.0,
        sources: recent.iter().map(|t| t.source.clone()).collect(),
        record_ids: recent.iter().map(|t| t.record_id.clone()).collect(),
        contested: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::Record;

    fn rec(seq: u64, ts: &str, body: Body) -> Stored {
        Stored {
            id: format!("r{seq}"),
            seq,
            record: Record { id: None, source: "bank:dkb".into(), ts: ts.parse().unwrap(), subject: None, tags: vec![], body },
        }
    }

    fn tx(seq: u64, ts: &str, amount: f64, who: &str) -> Stored {
        rec(
            seq,
            ts,
            Body::Transaction {
                account: "dkb".into(),
                amount,
                currency: "EUR".into(),
                counterparty: Some(who.into()),
                memo: None,
                category: None,
            },
        )
    }

    #[test]
    fn statement_plus_later_bookings() {
        let mut l = Ledger::default();
        l.add(&rec(1, "2026-09-01T00:00:00Z", Body::Balance { account: "dkb".into(), amount: 1000.0, currency: "EUR".into(), class: Some("cash".into()) }));
        l.add(&tx(2, "2026-09-10T00:00:00Z", -50.0, "REWE Markt"));
        l.add(&tx(3, "2026-09-20T00:00:00Z", -30.0, "REWE Markt"));
        let now = "2026-09-26T00:00:00Z".parse().unwrap();
        let c = l.candidates(&crate::terms::query_terms("bank balance"), now);
        let stmt = c.iter().find(|c| c.text.starts_with("account dkb (cash)")).unwrap();
        assert!(stmt.text.contains("statement + since = +920.00 EUR"), "{}", stmt.text);

        let rewe = l.candidates(&crate::terms::query_terms("how much at rewe"), now);
        let hit = rewe.iter().find(|c| c.text.contains("'rewe'")).unwrap();
        assert!(hit.text.contains("2 totalling -80.00 EUR"), "{}", hit.text);
    }
}
