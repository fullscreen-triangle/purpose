import { NextRequest, NextResponse } from "next/server";
import { createProfile, listProfiles } from "@/lib/profile-store";
import { slugify } from "@/lib/types";

export async function GET() {
  const profiles = await listProfiles();
  return NextResponse.json(profiles);
}

export async function POST(req: NextRequest) {
  const body = (await req.json()) as { displayName?: string };
  const displayName = body.displayName?.trim();
  if (!displayName) {
    return NextResponse.json({ error: "displayName is required" }, { status: 400 });
  }

  const id = slugify(displayName);
  if (!id) {
    return NextResponse.json(
      { error: "displayName must contain at least one letter or digit" },
      { status: 400 },
    );
  }

  const existing = await listProfiles();
  if (existing.some((p) => p.id === id)) {
    return NextResponse.json({ error: `a profile named '${id}' already exists` }, { status: 409 });
  }

  const profile = await createProfile(id, displayName);
  return NextResponse.json(profile, { status: 201 });
}
