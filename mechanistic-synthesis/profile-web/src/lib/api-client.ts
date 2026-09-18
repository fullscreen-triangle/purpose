// Thin client-side fetch wrappers for the API routes under src/app/api.
// Runs in the browser — no Node APIs here.

import type { ModelChoice, Profile } from "./types.js";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { error?: string };
    throw new Error(body.error ?? `request failed with status ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function listProfiles(): Promise<Profile[]> {
  return fetch("/api/profiles").then((r) => json(r));
}

export function createProfile(displayName: string): Promise<Profile> {
  return fetch("/api/profiles", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ displayName }),
  }).then((r) => json(r));
}

export function getProfile(id: string): Promise<Profile> {
  return fetch(`/api/profiles/${encodeURIComponent(id)}`).then((r) => json(r));
}

export function deleteProfile(id: string): Promise<void> {
  return fetch(`/api/profiles/${encodeURIComponent(id)}`, { method: "DELETE" }).then((r) => {
    if (!r.ok) throw new Error(`delete failed with status ${r.status}`);
  });
}

export function setProfileModel(id: string, model: ModelChoice): Promise<Profile> {
  return fetch(`/api/profiles/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  }).then((r) => json(r));
}

export interface UploadResult {
  accepted: string[];
  rejected: { filename: string; reason: string }[];
  profile: Profile;
}

export function uploadSources(id: string, files: FileList | File[]): Promise<UploadResult> {
  const form = new FormData();
  for (const file of Array.from(files)) {
    form.append("files", file);
  }
  return fetch(`/api/profiles/${encodeURIComponent(id)}/sources/upload`, {
    method: "POST",
    body: form,
  }).then((r) => json(r));
}

export function removeSourceFile(id: string, filename: string): Promise<Profile> {
  return fetch(
    `/api/profiles/${encodeURIComponent(id)}/sources/${encodeURIComponent(filename)}`,
    { method: "DELETE" },
  ).then((r) => json(r));
}

export function addSourceLink(id: string, url: string): Promise<Profile> {
  return fetch(`/api/profiles/${encodeURIComponent(id)}/sources/link`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  }).then((r) => json(r));
}

export function removeSourceLink(id: string, url: string): Promise<Profile> {
  return fetch(
    `/api/profiles/${encodeURIComponent(id)}/sources/link?url=${encodeURIComponent(url)}`,
    { method: "DELETE" },
  ).then((r) => json(r));
}

export function triggerBuild(id: string): Promise<Profile> {
  return fetch(`/api/profiles/${encodeURIComponent(id)}/build`, { method: "POST" }).then((r) =>
    json(r),
  );
}
