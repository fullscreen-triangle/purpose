import { mkdir, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { NextRequest, NextResponse } from "next/server";
import { loadProfile, ProfileNotFoundError, saveProfile, sourcesDir } from "@/lib/profile-store";
import { isSupportedExtension } from "@/lib/types";

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
    await writeFile(join(dir, filename), bytes);

    profile.sources.files = profile.sources.files.filter((f) => f.filename !== filename);
    profile.sources.files.push({
      filename,
      addedAt: new Date().toISOString(),
      sizeBytes: bytes.byteLength,
    });
    accepted.push(filename);
  }

  await saveProfile(profile);

  return NextResponse.json({ accepted, rejected, profile });
}
