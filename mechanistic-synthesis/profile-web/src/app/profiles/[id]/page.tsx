"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import {
  addSourceLink,
  getProfile,
  removeSourceFile,
  removeSourceLink,
  setProfileModel,
  triggerBuild,
  uploadSources,
} from "@/lib/api-client";
import type { ModelChoice, Profile } from "@/lib/types";
import { SUPPORTED_EXTENSIONS } from "@/lib/types";

export default function ProfileDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [profile, setProfile] = useState<Profile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [linkUrl, setLinkUrl] = useState("");

  function refresh() {
    getProfile(id).then(setProfile).catch((e: Error) => setError(e.message));
  }

  useEffect(refresh, [id]);

  async function handleUpload(files: FileList | null) {
    if (!files || files.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      const result = await uploadSources(id, files);
      setProfile(result.profile);
      if (result.rejected.length > 0) {
        setError(result.rejected.map((r) => `${r.filename}: ${r.reason}`).join("; "));
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handleAddLink(e: React.FormEvent) {
    e.preventDefault();
    if (!linkUrl.trim()) return;
    setBusy(true);
    setError(null);
    try {
      setProfile(await addSourceLink(id, linkUrl.trim()));
      setLinkUrl("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleRemoveFile(filename: string) {
    setBusy(true);
    try {
      setProfile(await removeSourceFile(id, filename));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleRemoveLink(url: string) {
    setBusy(true);
    try {
      setProfile(await removeSourceLink(id, url));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleModelChange(model: ModelChoice) {
    setBusy(true);
    try {
      setProfile(await setProfileModel(id, model));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleTrain() {
    setBusy(true);
    setError(null);
    // Optimistic "running" state — refreshed for real once the (blocking)
    // build call resolves.
    setProfile((p) => (p ? { ...p, lastBuild: { startedAt: new Date().toISOString(), status: "running" } } : p));
    try {
      setProfile(await triggerBuild(id));
    } catch (e) {
      setError((e as Error).message);
      refresh();
    } finally {
      setBusy(false);
    }
  }

  if (!profile) {
    return (
      <main>
        <Link href="/" className="text-sm text-primary hover:underline">
          ← Profiles
        </Link>
        <p className="mt-4 text-sm opacity-60">{error ?? "Loading…"}</p>
      </main>
    );
  }

  const hasSources = profile.sources.files.length > 0 || profile.sources.links.length > 0;
  const training = profile.lastBuild?.status === "running";

  return (
    <main>
      <Link href="/" className="text-sm text-primary hover:underline">
        ← Profiles
      </Link>
      <h1 className="mt-2 text-2xl font-semibold">{profile.displayName}</h1>

      {error && (
        <p className="mt-3 rounded border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-500">
          {error}
        </p>
      )}

      <section className="mt-6">
        <h2 className="text-sm font-semibold uppercase tracking-wide opacity-60">Sources</h2>

        <div className="mt-3 space-y-2">
          {profile.sources.files.map((f) => (
            <div key={f.filename} className="flex items-center justify-between rounded border border-dark/10 px-3 py-2 text-sm dark:border-light/10">
              <span>{f.filename}</span>
              <button onClick={() => handleRemoveFile(f.filename)} disabled={busy} className="text-xs opacity-60 hover:text-red-500 hover:opacity-100">
                remove
              </button>
            </div>
          ))}
          {profile.sources.links.map((l) => (
            <div key={l.url} className="flex items-center justify-between rounded border border-dark/10 px-3 py-2 text-sm dark:border-light/10">
              <span className="truncate">{l.url}</span>
              <button onClick={() => handleRemoveLink(l.url)} disabled={busy} className="ml-2 shrink-0 text-xs opacity-60 hover:text-red-500 hover:opacity-100">
                remove
              </button>
            </div>
          ))}
          {!hasSources && <p className="text-sm opacity-50">No sources yet.</p>}
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-3">
          <label className="cursor-pointer rounded border border-dark/20 px-3 py-2 text-sm dark:border-light/20">
            Add files ({SUPPORTED_EXTENSIONS.join(", ")})
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept={SUPPORTED_EXTENSIONS.map((e) => `.${e}`).join(",")}
              onChange={(e) => handleUpload(e.target.files)}
              disabled={busy}
              className="hidden"
            />
          </label>

          <form onSubmit={handleAddLink} className="flex flex-1 gap-2">
            <input
              value={linkUrl}
              onChange={(e) => setLinkUrl(e.target.value)}
              placeholder="https://…"
              className="flex-1 rounded border border-dark/20 bg-transparent px-3 py-2 text-sm outline-none focus:border-primary dark:border-light/20"
            />
            <button
              type="submit"
              disabled={busy || !linkUrl.trim()}
              className="rounded border border-dark/20 px-3 py-2 text-sm dark:border-light/20"
            >
              Add link
            </button>
          </form>
        </div>
      </section>

      <section className="mt-8">
        <h2 className="text-sm font-semibold uppercase tracking-wide opacity-60">Model</h2>
        <ModelPicker model={profile.model} onChange={handleModelChange} disabled={busy} />
      </section>

      <section className="mt-8">
        <button
          onClick={handleTrain}
          disabled={busy || training || !hasSources}
          className="rounded bg-primary px-5 py-2.5 text-sm font-medium text-white disabled:opacity-50"
        >
          {training ? "Training…" : "Train"}
        </button>
        {!hasSources && <p className="mt-2 text-xs opacity-50">Add at least one source to train.</p>}

        <BuildStatus profile={profile} />
      </section>
    </main>
  );
}

function ModelPicker({
  model,
  onChange,
  disabled,
}: {
  model: ModelChoice;
  onChange: (m: ModelChoice) => void;
  disabled: boolean;
}) {
  const [repo, setRepo] = useState(model.kind === "pretrained" ? model.repo : "");

  return (
    <div className="mt-3 space-y-2">
      <label className="flex items-center gap-2 text-sm">
        <input
          type="radio"
          checked={model.kind === "scratch"}
          disabled={disabled}
          onChange={() => onChange({ kind: "scratch" })}
        />
        Train from scratch on this profile&apos;s corpus (fast, no network)
      </label>
      <label className="flex items-center gap-2 text-sm">
        <input
          type="radio"
          checked={model.kind === "pretrained"}
          disabled={disabled}
          onChange={() => repo.trim() && onChange({ kind: "pretrained", repo: repo.trim() })}
        />
        LoRA-adapt a pretrained HuggingFace checkpoint
      </label>
      {model.kind === "pretrained" && (
        <input
          value={repo}
          onChange={(e) => setRepo(e.target.value)}
          onBlur={() => repo.trim() && onChange({ kind: "pretrained", repo: repo.trim() })}
          placeholder="owner/name, e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0"
          disabled={disabled}
          className="ml-6 w-full max-w-sm rounded border border-dark/20 bg-transparent px-3 py-1.5 text-sm outline-none focus:border-primary dark:border-light/20"
        />
      )}
    </div>
  );
}

function BuildStatus({ profile }: { profile: Profile }) {
  const build = profile.lastBuild;
  if (!build) return null;

  if (build.status === "running") {
    return <p className="mt-4 text-sm opacity-70">Training in progress — this may take a while.</p>;
  }
  if (build.status === "error") {
    return (
      <pre className="mt-4 whitespace-pre-wrap rounded border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-500">
        {build.error}
      </pre>
    );
  }
  if (build.status === "done" && build.result) {
    return (
      <div className="mt-4 rounded border border-emerald-500/30 bg-emerald-500/10 p-3 text-sm">
        <p className="font-medium text-emerald-600 dark:text-emerald-400">Model ready</p>
        <p className="mt-1 text-xs opacity-70">
          {build.result.documentCount} document(s), {build.result.exampleCount} example(s), vocab{" "}
          {build.result.vocabSize}
        </p>
        <p className="mt-1 break-all text-xs opacity-50">{build.result.path}</p>
      </div>
    );
  }
  return null;
}
