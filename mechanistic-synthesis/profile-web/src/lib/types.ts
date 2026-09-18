// Shared types for a profile — the frontend's own concept, layered over a
// purpose-factory theme.toml. A profile has no Rust-side representation;
// theme-toml.ts turns one into the theme.toml purpose-factory understands.

export type ModelChoice =
  | { kind: "scratch" }
  | { kind: "pretrained"; repo: string; revision?: string };

export interface SourceFile {
  filename: string;
  addedAt: string;
  sizeBytes: number;
}

export interface SourceLink {
  url: string;
  addedAt: string;
}

export type BuildStatus = "running" | "done" | "error";

export interface BuildRecord {
  startedAt: string;
  finishedAt?: string;
  status: BuildStatus;
  error?: string;
  /** The ThemeModel result, once status is "done". */
  result?: {
    path: string;
    documentCount: number;
    exampleCount: number;
    vocabSize: number;
  };
}

export interface Profile {
  id: string;
  displayName: string;
  createdAt: string;
  sources: {
    files: SourceFile[];
    links: SourceLink[];
  };
  model: ModelChoice;
  lastBuild?: BuildRecord;
}

/** Extensions purpose-factory's LocalFileSource knows how to ingest. */
export const SUPPORTED_EXTENSIONS = ["tex", "pdf", "md", "txt", "csv", "json"] as const;
export type SupportedExtension = (typeof SUPPORTED_EXTENSIONS)[number];

export function isSupportedExtension(filename: string): boolean {
  const ext = filename.split(".").pop()?.toLowerCase();
  return !!ext && (SUPPORTED_EXTENSIONS as readonly string[]).includes(ext);
}

/** URL-safe, filesystem-safe slug for a profile id. */
export function slugify(name: string): string {
  return name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 64);
}
