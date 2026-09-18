import { mkdir, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { NextRequest, NextResponse } from "next/server";
import { http } from "@buhera/purpose-factory-client";
import { loadProfile, ProfileNotFoundError, saveProfile, sourcesDir } from "@/lib/profile-store";
import { isSupportedExtension } from "@/lib/types";
import { remoteServerConfig } from "@/lib/remote-config";

const MAX_UPLOAD_BYTES = 25 * 1024 * 1024; // 25MB per file

/** Strips any directory components and disallows empty/dotfile-only names,
 * so an uploaded filename can never escape the profile's sources directory. */
function sanitizeFilename(name: string): string | null {
  const base = basename(name.replace(/\\/g, "/"));
  if (!base || base === "." || base === "..") return null;
  return base;
}

export async function POST(req: NextRequest, { params }: { params: { id: string } }) {
  let profile;
  try {
    profile = await loadProfile(params.id);
  } catch (err) {
    if (err instanceof ProfileNotFoundError) {
      return NextResponse.json({ error: err.message }, { status: 404 });
    }
    throw err;
  }

  const form = await req.formData();
  const files = form.getAll("files").filter((f): f is File => f instanceof File);
  if (files.length === 0) {
    return NextResponse.json({ error: "no files in 'files' field" }, { status: 400 });
  }

  const dir = sourcesDir(profile.id);
  await mkdir(dir, { recursive: true });

  const accepted: string[] = [];
  const rejected: { filename: string; reason: string }[] = [];
  const validated: { filename: string; bytes: Buffer }[] = [];

  for (const file of files) {
    const filename = sanitizeFilename(file.name);
    if (!filename) {
      rejected.push({ filename: file.name, reason: "invalid filename" });
      continue;
    }
    if (!isSupportedExtension(filename)) {
      rejected.push({ filename, reason: "unsupported extension (supported: tex, pdf, md, txt, csv, json)" });
      continue;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      rejected.push({ filename, reason: `exceeds ${MAX_UPLOAD_BYTES / (1024 * 1024)}MB limit` });
      continue;
    }

    const bytes = Buffer.from(await file.arrayBuffer());
    validated.push({ filename, bytes });
  }

  // Local copy always stays the source of truth for the profile UI's own
  // file list, whether or not a remote server is configured.
  for (const { filename, bytes } of validated) {
    await writeFile(join(dir, filename), bytes);
    profile.sources.files = profile.sources.files.filter((f) => f.filename !== filename);
    profile.sources.files.push({
      filename,
      addedAt: new Date().toISOString(),
      sizeBytes: bytes.byteLength,
    });
    accepted.push(filename);
  }

  const remote = remoteServerConfig();
  if (remote && validated.length > 0) {
    const remoteResult = await http.uploadSources(
      profile.id,
      validated.map((v) => ({ filename: v.filename, data: v.bytes })),
      remote,
    );
    // Anything the remote server rejects (e.g. a stricter future check)
    // is surfaced too, even though the local write already succeeded —
    // the profile's file list reflects what's usable for training, which
    // in remote mode means what the server accepted.
    for (const r of remoteResult.rejected) {
      rejected.push({ filename: r.filename, reason: `remote server: ${r.reason}` });
    }
  }

  await saveProfile(profile);

  return NextResponse.json({ accepted, rejected, profile });
}
