// Filesystem persistence for profiles. One JSON file per profile under
// PROFILES_ROOT/<id>/profile.json, sibling to PROFILES_ROOT/<id>/sources/
// where uploaded files land — that sources/ directory is exactly the
// local_files.root a generated theme.toml points purpose-factory at.

import { mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { Profile } from "./types.js";

// Rooted next to the Rust workspace so it sits alongside .purpose/ the same
// way the CLI's own state does, rather than inside this Next.js app.
export const PROFILES_ROOT = join(process.cwd(), "..", "implementation", ".purpose-profiles");

function profileDir(id: string): string {
  return join(PROFILES_ROOT, id);
}

export function sourcesDir(id: string): string {
  return join(profileDir(id), "sources");
}

function profileJsonPath(id: string): string {
  return join(profileDir(id), "profile.json");
}

export class ProfileNotFoundError extends Error {
  constructor(id: string) {
    super(`profile '${id}' not found`);
    this.name = "ProfileNotFoundError";
  }
}

export async function listProfiles(): Promise<Profile[]> {
  await mkdir(PROFILES_ROOT, { recursive: true });
  const entries = await readdir(PROFILES_ROOT, { withFileTypes: true });
  const profiles: Profile[] = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    try {
      profiles.push(await loadProfile(entry.name));
    } catch {
      // Skip a directory that isn't a valid profile (no profile.json, or
      // corrupt JSON) rather than failing the whole list.
    }
  }
  profiles.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  return profiles;
}

export async function loadProfile(id: string): Promise<Profile> {
  try {
    const raw = await readFile(profileJsonPath(id), "utf8");
    return JSON.parse(raw) as Profile;
  } catch (err) {
    if (isNoEnt(err)) {
      throw new ProfileNotFoundError(id);
    }
    throw err;
  }
}

export async function createProfile(id: string, displayName: string): Promise<Profile> {
  await mkdir(sourcesDir(id), { recursive: true });
  const profile: Profile = {
    id,
    displayName,
    createdAt: new Date().toISOString(),
    sources: { files: [], links: [] },
    model: { kind: "scratch" },
  };
  await saveProfile(profile);
  return profile;
}

export async function saveProfile(profile: Profile): Promise<void> {
  await mkdir(profileDir(profile.id), { recursive: true });
  await writeFile(profileJsonPath(profile.id), JSON.stringify(profile, null, 2));
}

export async function deleteProfile(id: string): Promise<void> {
  await rm(profileDir(id), { recursive: true, force: true });
}

function isNoEnt(err: unknown): boolean {
  return (
    typeof err === "object" &&
    err !== null &&
    "code" in err &&
    (err as { code?: unknown }).code === "ENOENT"
  );
}
