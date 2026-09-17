import { NextRequest, NextResponse } from "next/server";
import { deleteProfile, loadProfile, ProfileNotFoundError, saveProfile } from "@/lib/profile-store";
import type { ModelChoice } from "@/lib/types";

export async function GET(_req: NextRequest, { params }: { params: { id: string } }) {
  try {
    const profile = await loadProfile(params.id);
    return NextResponse.json(profile);
  } catch (err) {
    if (err instanceof ProfileNotFoundError) {
      return NextResponse.json({ error: err.message }, { status: 404 });
    }
    throw err;
  }
}

/** Updates the profile's model choice (scratch vs. pretrained repo). */
export async function PATCH(req: NextRequest, { params }: { params: { id: string } }) {
  try {
    const profile = await loadProfile(params.id);
    const body = (await req.json()) as { model?: ModelChoice };
    if (body.model) {
      profile.model = body.model;
    }
    await saveProfile(profile);
    return NextResponse.json(profile);
  } catch (err) {
    if (err instanceof ProfileNotFoundError) {
      return NextResponse.json({ error: err.message }, { status: 404 });
    }
    throw err;
  }
}

export async function DELETE(_req: NextRequest, { params }: { params: { id: string } }) {
  await deleteProfile(params.id);
  return NextResponse.json({ ok: true });
}
