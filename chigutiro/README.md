# chigutiro

A personal model that learns continuously from whatever its host framework
feeds it — wearables, phone sensors, email, documents, contacts, plans,
finances, athletics logs — on two timescales:

- **Fast (every record).** Each record is committed to an append-only,
  encrypted log and indexed by a typed receiver before `ingest` returns, so the
  next question already sees it. Questions are answered from *claims* those
  receivers compute at query time.
- **Slow (rounds).** Your own prose accumulates as a redacted *voice corpus*.
  A consolidation round retrains a LoRA adapter on it from the base checkpoint
  via `purpose factory build`, then deletes every older model.

It is a self-contained Rust service with a TypeScript bridge. The host (for
example kwisatz-haderach) pushes records in and asks questions; chigutiro
knows nothing about the host.

## What reaches the weights, and what does not

Only prose **you wrote** (`authored_by_owner: true`) can reach model weights,
after quoted replies and signatures are cut and addresses, IBANs, card
numbers, phone numbers, one-time codes, credentials and secret-bearing URLs
are redacted. Everything else stays in receivers:

| Record `kind` | Receiver | Answers with |
|---|---|---|
| `prose` | `text` (BM25 over passages) | passages, dated and sourced |
| `measurement` | `series` | latest value and its age, 7-day mean against the prior 7 days, 28-day mean; flags **contested** when sources disagree on the same day |
| `transaction`, `balance` | `ledger` | latest statement plus later bookings, 30-day flows, spend by payee; never sums across currencies |
| `position` | `places` | last fix, labelled places by visit count |
| `contact` | `people` | person, role, organisation, how often other records reference them |
| `event` | `calendar` | matching and upcoming events, relative to now |

Weights can't be asked for a current value, can't be kept current, and can't
forget one person. Receivers can do all three.

## Scopes: what of the profile is work

A chigutiro profile is one person's whole life. In absicht it is that
person's account, and only a narrow **work** slice of it ever takes part:
the slice an absicht federation may consult, and the only material that may
be trained into a model off this machine (e.g. on AppHub). Everything else is
**personal** and never leaves.

The scope is decided by the record's **source channel**, never by its
content, and is worked out when a record is read, so narrowing the policy
applies at once to everything already in the log. For now, work is exactly:

| Channel (`source`) | What it is |
|---|---|
| `chat:<app>` | chat session history with an assistant |
| `academic:<engine>` | academic searches and conversations about them |
| `upload:lab-report`, `upload:paper`, `upload:presentation` | documents the user uploaded (extracted text) |

Only prose counts. A measurement or a contact is personal whatever its
source. University email, notes and everything else stay personal until
their channel is added (`--work-channels` / `CHIGUTIRO_WORK_CHANNELS`). The
host must label channels honestly; the bridge's `chatSession`, `academic` and
`upload` adapters do.

- **Ask the work view:** `POST /ask { query, scope: "work" }`, or
  `chigutiro ask --work "…"`. It searches work records only. There is no
  personal-only view.
- **Export the work corpus:** `chigutiro export-work --out DIR` writes
  `corpus.jsonl` (`{text, source}` per document) and `manifest.json` (counts
  per channel, record ids). It is redacted like the voice corpus, but unlike
  it, it includes text others wrote, such as a paper's authors or an
  assistant's replies. That is right for a model of the field, and why it is
  a separate corpus from the voice corpus. It is plaintext: ship it, then
  delete it.

## Answers are graded

A claim's grade is the number of independent sources behind it:
`single_sourced`, `two_sourced`, and `grounded` only at three or more. If any
sources disagree it is `contested`, and with no claims the answer is
`declined`. An answer's grade is its weakest admitted claim. Resting HR from
Garmin, Polar and a manual log is `grounded`; one email is `single_sourced`.

Claims compete for a fixed budget by water-filling. Each receiver's
candidates have non-increasing gains (`coverage · decay^i`), and the budget
goes to the largest marginal gains. A dense receiver can't crowd out the
others, and the gain of the last admitted claim is the clearing price. Each
route is logged to `routes.log` as its **shape** only: receivers, offered and
admitted counts, price, grade. It never contains the query or any content.

If `CHIGUTIRO_OLLAMA_MODEL` is set, the claims are phrased by that model,
which is instructed to use nothing else. Otherwise, or if it fails, the
answer is the graded claims themselves.

## Erasure

`erase { subject | ids | source | before }` removes matching records from
every receiver and rewrites the log without them, so they are gone from disk,
not tombstoned. If voice material was removed, the current model is reported
`tainted` and a round becomes due. When that round succeeds, every older
model is deleted. Use a third party's email address as the record `subject`
and erasing it removes that person completely.

## Run

```bash
cargo build --release
./target/release/chigutiro keygen            # prints CHIGUTIRO_KEY and CHIGUTIRO_TOKEN
export CHIGUTIRO_KEY=... CHIGUTIRO_TOKEN=...
./target/release/chigutiro serve --data ./data   # http://127.0.0.1:8740
```

| Env | Meaning |
|---|---|
| `CHIGUTIRO_KEY` | 32-byte hex key for the record log (XChaCha20-Poly1305). Required unless `--plaintext`. **Losing it loses the log.** |
| `CHIGUTIRO_TOKEN` | Bearer token for every route but `/health`. |
| `CHIGUTIRO_DATA`, `CHIGUTIRO_HOST`, `CHIGUTIRO_PORT` | Data dir (default `.chigutiro`), bind address (default loopback), port (8740). |
| `CHIGUTIRO_OLLAMA_URL`, `CHIGUTIRO_OLLAMA_MODEL`, `CHIGUTIRO_OWNER` | Optional phrasing model, and how it names you. |
| `CHIGUTIRO_PURPOSE_BIN` | A `purpose` binary with the `factory` subcommand; enables consolidation. |
| `CHIGUTIRO_BASE_MODEL` | Checkpoint each round retrains from (default `Qwen/Qwen2.5-0.5B-Instruct`; must be a single unsharded `model.safetensors`). |
| `CHIGUTIRO_MIN_NEW_DOCS`, `CHIGUTIRO_AUTO_CONSOLIDATE` | New voice documents that make a round due (50); start rounds automatically after ingest. |
| `CHIGUTIRO_WORK_CHANNELS` | Source channels that count as work, comma-separated (default `chat,academic,upload:lab-report,upload:paper,upload:presentation`). |

CLI: `ingest FILE|-`, `ask [--work] "…"`, `export-work --out DIR`, `status`, `erase --subject …`,
`consolidate [--force]`, and `verify`. `verify` checks the log decodes, ids
are unique, sequence numbers increase, the committed count covers every
record, superseded models are deleted and no plaintext corpus is left behind.
It exits 1 on any breach.

## HTTP API

All routes except `/health` take `Authorization: Bearer $CHIGUTIRO_TOKEN`.
There is no CORS: call chigutiro from a server, never from a browser.

| Route | Body → Response |
|---|---|
| `GET /health` | `{ ok, version }` |
| `GET /status` | counts per receiver, voice corpus size, consolidation state |
| `POST /ingest` | `{ records: [...] }` → `{ accepted, duplicates, rejected: [{index, reason}], committed }` |
| `POST /ask` | `{ query, budget?, generate?, scope? }` (`scope: "work"`: work records only) → `{ answer, generated, grade, claims, route, model_version, model_tainted }` |
| `POST /erase` | `{ ids?, subject?, source?, before? }` (all given must match) → `{ removed, voice_material_removed, model_tainted }` |
| `POST /consolidate` | `{ force? }` → 202 `{ started, version }` or 409 `{ started: false, reason }` |
| `GET /consolidations` | round history |

Re-sending a record is a no-op. Identity is `source:id` if you give an
`id`, otherwise a hash of the content.

## TypeScript bridge (`bridge-ts`, `@buhera/chigutiro`)

```ts
import { ChigutiroClient, fromGarminSummary, fromBankTransactions, fromGmailMessages, measurements, event } from "@buhera/chigutiro";

const chigutiro = new ChigutiroClient({ baseUrl: process.env.CHIGUTIRO_URL!, token: process.env.CHIGUTIRO_TOKEN! });

// e.g. in a kwisatz-haderach Next.js API route or a scheduled job:
const garmin = await fetch(`${BACKEND}/health/summary`).then((r) => r.json());
await chigutiro.ingest([
  ...fromGarminSummary(garmin, yesterday),
  ...fromBankTransactions((await fetch(`${BACKEND}/bank/transactions?limit=500`).then((r) => r.json())).transactions).records,
  ...fromGmailMessages(await fetchRecentEmails(), ["you@example.org"]),
  ...measurements("track-log", sessionStart, { "100m time": [11.42, "s"], "top speed": [9.8, "m/s"] }, sessionId),
  event("plans", "2027-03-02", "Zanzibar", { end: "2027-03-16", place: "Stone Town" }),
]);

const { answer, grade, claims } = await chigutiro.ask("how is my HRV trending");
```

The adapters map the shapes kwisatz-haderach already produces: backend
`/health/summary`, `/bank/transactions`, and `web/lib/gmail.js`. The generic
constructors (`measurement(s)`, `event`, `contact`, `position`, `balance`)
cover any other source. `npm test` includes an end-to-end test against the
built binary.

## Deploy

`Dockerfile` builds the API server. Mount a persistent volume at `/data`,
supply `CHIGUTIRO_KEY` and `CHIGUTIRO_TOKEN` as secrets, and keep the port on
a private network or behind TLS. Hosts whose disk is ephemeral lose the log
on restart. Consolidation is training, so run it where there is compute, not
on a small web instance.

## Limits, stated plainly

- Receivers match on lexical terms. "How am I recovering?" finds nothing
  unless a metric or passage uses those words. Compiling intent into receiver
  queries is the resolver's job, and that resolver is not built yet.
- The contested tolerance (15%) is a declared constant, not the derived
  partition-extinction lock of the route-graph paper. It is labelled as
  declared in every claim that uses it.
- Consolidation trains in F32 on CPU through `purpose factory`, which is slow
  for real base models. The exported safetensors are not automatically
  served: a `Modelfile` is written next to them, but Ollama's import of this
  architecture is untested here.
- The corpus exists on disk in plaintext only while a round runs, and the
  exported model is plaintext. Keep the data directory on an encrypted
  volume.
