import { rm } from "node:fs/promises";
import { join } from "node:path";
import { NextRequest, NextResponse } from "next/server";
import { loadProfile, ProfileNotFoundError, saveProfile, sourcesDir } from "@/lib/profile-store";

/** Removes an uploaded source file. The dynamic segment is a filename,
 * already sanitized to a bare basename at upload time — decoded here since
 * Next.js URL-encodes route params. */
export async function DELETE(_req: NextRequest, { params }: { params: { id: string; filename: string } }) {
  let profile;
  try {
    profile = await loadProfile(params.id);
  } catch (err) {
    if (err instanceof ProfileNotFoundError) {
      return NextResponse.json({ error: err.message }, { status: 404 });
    }
    throw err;
  }

  const filename = decodeURIComponent(params.filename);
  const existed = profile.sources.files.some((f) => f.filename === filename);
  if (!existed) {
    return NextResponse.json({ error: `no such source file '${filename}'` }, { status: 404 });
  }

  await rm(join(sourcesDir(profile.id), filename), { force: true });
  profile.sources.files = profile.sources.files.filter((f) => f.filename !== filename);
  await saveProfile(profile);

  return NextResponse.json(profile);
}
