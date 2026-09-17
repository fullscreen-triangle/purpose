"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { createProfile, listProfiles } from "@/lib/api-client";
import type { Profile } from "@/lib/types";

export default function ProfileListPage() {
  const [profiles, setProfiles] = useState<Profile[] | null>(null);
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listProfiles().then(setProfiles).catch((e: Error) => setError(e.message));
  }, []);

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!newName.trim()) return;
    setCreating(true);
    setError(null);
    try {
      const profile = await createProfile(newName.trim());
      setProfiles((prev) => (prev ? [profile, ...prev] : [profile]));
      setNewName("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setCreating(false);
    }
  }

  return (
    <main>
      <h1 className="text-2xl font-semibold">Profiles</h1>
      <p className="mt-1 text-sm opacity-70">
        A profile is a personal knowledge theme: add papers, CSVs, JSON, and links, then train a
        model from it.
      </p>

      <form onSubmit={handleCreate} className="mt-6 flex gap-2">
        <input
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          placeholder="New profile name"
          className="flex-1 rounded border border-dark/20 bg-transparent px-3 py-2 text-sm outline-none focus:border-primary dark:border-light/20"
        />
        <button
          type="submit"
          disabled={creating || !newName.trim()}
          className="rounded bg-primary px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
        >
          {creating ? "Creating…" : "Create"}
        </button>
      </form>
      {error && <p className="mt-2 text-sm text-red-500">{error}</p>}

      <ul className="mt-8 space-y-3">
        {profiles === null && <li className="text-sm opacity-60">Loading…</li>}
        {profiles?.length === 0 && (
          <li className="text-sm opacity-60">No profiles yet — create one above.</li>
        )}
        {profiles?.map((p) => (
          <li key={p.id}>
            <Link
              href={`/profiles/${p.id}`}
              className="block rounded border border-dark/10 p-4 transition hover:border-primary dark:border-light/10"
            >
              <div className="flex items-center justify-between">
                <span className="font-medium">{p.displayName}</span>
                <StatusBadge profile={p} />
              </div>
              <p className="mt-1 text-xs opacity-60">
                {p.sources.files.length} file(s), {p.sources.links.length} link(s)
              </p>
            </Link>
          </li>
        ))}
      </ul>
    </main>
  );
}

function StatusBadge({ profile }: { profile: Profile }) {
  const status = profile.lastBuild?.status;
  if (!status) {
    return <span className="text-xs opacity-50">not trained</span>;
  }
  const styles: Record<string, string> = {
    running: "text-amber-500",
    done: "text-emerald-500",
    error: "text-red-500",
  };
  return <span className={`text-xs ${styles[status]}`}>{status}</span>;
}
