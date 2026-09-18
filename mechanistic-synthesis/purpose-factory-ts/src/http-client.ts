// HTTP client for a `purpose serve` server (crates/purpose-factory/src/server.rs),
// authenticated with a static bearer token. Same public shape as
// cli-bridge.ts (buildTheme, listThemes) so a caller can swap between the
// subprocess and network transports without changing call sites, plus
// uploadSources/downloadModelFile — new capabilities the subprocess bridge
// has no equivalent for, since a local subprocess already shares a
// filesystem with its caller.

import { fromRaw, type ThemeModel } from "./types.js";

export interface HttpClientOptions {
  /** Base URL of the `purpose serve` server, e.g. "http://localhost:8420". */
  baseUrl: string;
  /** Bearer token — must match the server's PURPOSE_SERVE_TOKEN. */
  token: string;
}

export class HttpClientError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "HttpClientError";
  }
}

function authHeaders(opts: HttpClientOptions): Record<string, string> {
  return { Authorization: `Bearer ${opts.token}` };
}

async function readErrorMessage(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { error?: string };
    return body.error ?? `request failed with status ${res.status}`;
  } catch {
    return `request failed with status ${res.status}`;
  }
}

/** POSTs `<baseUrl>/themes/<name>/sources` as multipart form data. */
export async function uploadSources(
  name: string,
  files: { filename: string; data: Blob | Buffer }[],
  opts: HttpClientOptions,
): Promise<{ accepted: string[]; rejected: { filename: string; reason: string }[] }> {
  const form = new FormData();
  for (const file of files) {
    const blob =
      file.data instanceof Blob ? file.data : new Blob([new Uint8Array(file.data)]);
    form.append("files", blob, file.filename);
  }

  const res = await fetch(`${opts.baseUrl}/themes/${encodeURIComponent(name)}/sources`, {
    method: "POST",
    headers: authHeaders(opts),
    body: form,
  });
  if (!res.ok) {
    throw new HttpClientError(await readErrorMessage(res), res.status);
  }
  return res.json() as Promise<{
    accepted: string[];
    rejected: { filename: string; reason: string }[];
  }>;
}

export type RemoteModelChoice =
  | { kind: "scratch" }
  | { kind: "pretrained"; repo: string; revision?: string };

export interface RemoteTrainingSpec {
  epochs?: number;
  batchSize?: number;
  learningRate?: number;
  loraRank?: number;
  loraAlpha?: number;
}

/**
 * POSTs `<baseUrl>/themes/<name>/build` — sources must already be uploaded
 * (via `uploadSources`) or supplied as `urls`; the server trains and
 * returns the resulting `ThemeModel`. Resolves once training finishes, same
 * as `cli-bridge.ts`'s `buildTheme`.
 */
export async function buildTheme(
  name: string,
  options: {
    urls?: string[];
    model?: RemoteModelChoice;
    training?: RemoteTrainingSpec;
  },
  opts: HttpClientOptions,
): Promise<ThemeModel> {
  const body = {
    urls: options.urls ?? [],
    model:
      options.model?.kind === "pretrained"
        ? { kind: "pretrained", repo: options.model.repo, revision: options.model.revision }
        : { kind: "scratch" },
    training: options.training
      ? {
          epochs: options.training.epochs,
          batch_size: options.training.batchSize,
          learning_rate: options.training.learningRate,
          lora_rank: options.training.loraRank,
          lora_alpha: options.training.loraAlpha,
        }
      : undefined,
  };

  const res = await fetch(`${opts.baseUrl}/themes/${encodeURIComponent(name)}/build`, {
    method: "POST",
    headers: { ...authHeaders(opts), "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new HttpClientError(await readErrorMessage(res), res.status);
  }
  const raw = await res.json();
  return fromRaw(raw as Parameters<typeof fromRaw>[0]);
}

/** GETs `<baseUrl>/themes` — same shape as `purpose factory list --raw`. */
export async function listThemes(opts: HttpClientOptions): Promise<ThemeModel[]> {
  const res = await fetch(`${opts.baseUrl}/themes`, { headers: authHeaders(opts) });
  if (!res.ok) {
    throw new HttpClientError(await readErrorMessage(res), res.status);
  }
  const raw: unknown = await res.json();
  return Array.isArray(raw) ? raw.map(fromRaw) : [];
}

/** GETs one of a built theme's model files (model.safetensors, config.json,
 * tokenizer.json) as raw bytes. */
export async function downloadModelFile(
  name: string,
  file: "model.safetensors" | "config.json" | "tokenizer.json",
  opts: HttpClientOptions,
): Promise<ArrayBuffer> {
  const res = await fetch(
    `${opts.baseUrl}/themes/${encodeURIComponent(name)}/model/${encodeURIComponent(file)}`,
    { headers: authHeaders(opts) },
  );
  if (!res.ok) {
    throw new HttpClientError(await readErrorMessage(res), res.status);
  }
  return res.arrayBuffer();
}
