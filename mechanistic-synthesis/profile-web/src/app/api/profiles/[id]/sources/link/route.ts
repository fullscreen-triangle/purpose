import { NextRequest, NextResponse } from "next/server";
import { loadProfile, ProfileNotFoundError, saveProfile } from "@/lib/profile-store";

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

  const body = (await req.json()) as { url?: string };
  const url = body.url?.trim();
  if (!url) {
    return NextResponse.json({ error: "url is required" }, { status: 400 });
  }
  try {
    // eslint-disable-next-line no-new
    new URL(url);
  } catch {
    return NextResponse.json({ error: "url is not a valid absolute URL" }, { status: 400 });
  }

  if (!profile.sources.links.some((l) => l.url === url)) {
    profile.sources.links.push({ url, addedAt: new Date().toISOString() });
    await saveProfile(profile);
  }

  return NextResponse.json(profile);
}

export async function DELETE(req: NextRequest, { params }: { params: { id: string } }) {
  let profile;
  try {
    profile = await loadProfile(params.id);
  } catch (err) {
    if (err instanceof ProfileNotFoundError) {
      return NextResponse.json({ error: err.message }, { status: 404 });
    }
    throw err;
  }

  const url = req.nextUrl.searchParams.get("url");
  if (!url) {
    return NextResponse.json({ error: "url query parameter is required" }, { status: 400 });
  }

  profile.sources.links = profile.sources.links.filter((l) => l.url !== url);
  await saveProfile(profile);

  return NextResponse.json(profile);
}
