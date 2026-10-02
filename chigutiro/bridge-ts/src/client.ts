import type {
  Answer,
  ChigutiroRecord,
  Consolidation,
  EraseCriteria,
  EraseReport,
  IngestReport,
  Scope,
  Status,
} from "./types.js";

export interface ClientOptions {
  /** e.g. http://127.0.0.1:8740 */
  baseUrl: string;
  /** CHIGUTIRO_TOKEN. Server-side only: never ship this to a browser. */
  token: string;
  /** Per-request timeout; `ask` with generation can take tens of seconds. */
  timeoutMs?: number;
  /** Records per `/ingest` request. */
  batchSize?: number;
  fetch?: typeof fetch;
}

export class ChigutiroError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly body: unknown,
  ) {
    super(message);
    this.name = "ChigutiroError";
  }
}

export class ChigutiroClient {
  private readonly baseUrl: string;
  private readonly token: string;
  private readonly timeoutMs: number;
  private readonly batchSize: number;
  private readonly fetchImpl: typeof fetch;

  constructor(opts: ClientOptions) {
    if (!opts.token) throw new Error("chigutiro: token is required");
    this.baseUrl = opts.baseUrl.replace(/\/+$/, "");
    this.token = opts.token;
    this.timeoutMs = opts.timeoutMs ?? 120_000;
    this.batchSize = opts.batchSize ?? 500;
    this.fetchImpl = opts.fetch ?? globalThis.fetch;
  }

  private async call<T>(method: "GET" | "POST", path: string, body?: unknown, okStatuses = [200]): Promise<T> {
    const res = await this.fetchImpl(`${this.baseUrl}${path}`, {
      method,
      headers: {
        authorization: `Bearer ${this.token}`,
        ...(body === undefined ? {} : { "content-type": "application/json" }),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    const text = await res.text();
    let parsed: unknown = text;
    try {
      parsed = text ? JSON.parse(text) : null;
    } catch {
      // Keep the raw text for the error below.
    }
    if (!okStatuses.includes(res.status)) {
      const msg = (parsed as { error?: string; reason?: string } | null)?.error
        ?? (parsed as { reason?: string } | null)?.reason
        ?? String(text).slice(0, 200);
      throw new ChigutiroError(`chigutiro ${method} ${path}: ${res.status} ${msg}`, res.status, parsed);
    }
    return parsed as T;
  }

  health(): Promise<{ ok: boolean; version: string }> {
    return this.call("GET", "/health");
  }

  status(): Promise<Status> {
    return this.call("GET", "/status");
  }

  /** Sends records in batches; the report sums every batch. Rejected indices refer to `records`. */
  async ingest(records: ChigutiroRecord[]): Promise<IngestReport> {
    const total: IngestReport = { accepted: 0, duplicates: 0, rejected: [], committed: 0 };
    for (let start = 0; start < records.length; start += this.batchSize) {
      const batch = records.slice(start, start + this.batchSize);
      const r = await this.call<IngestReport>("POST", "/ingest", { records: batch });
      total.accepted += r.accepted;
      total.duplicates += r.duplicates;
      total.rejected.push(...r.rejected.map((x) => ({ ...x, index: x.index + start })));
      total.committed = r.committed;
    }
    return total;
  }

  ask(query: string, opts: { budget?: number; generate?: boolean; scope?: Scope } = {}): Promise<Answer> {
    return this.call("POST", "/ask", { query, ...opts });
  }

  erase(criteria: EraseCriteria): Promise<EraseReport> {
    return this.call("POST", "/erase", criteria);
  }

  /** Starts a round in the background. Resolves `started: false` with a reason if none may start. */
  consolidate(force = false): Promise<{ started: boolean; version?: number; reason?: string }> {
    return this.call("POST", "/consolidate", { force }, [202, 409]);
  }

  consolidations(): Promise<Consolidation[]> {
    return this.call("GET", "/consolidations");
  }
}
