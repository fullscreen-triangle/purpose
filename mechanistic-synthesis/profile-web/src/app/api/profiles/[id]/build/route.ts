import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { NextRequest, NextResponse } from "next/server";
import { cli, http, type ThemeModel } from "@buhera/purpose-factory-client";
import { loadProfile, PROFILES_ROOT, ProfileNotFoundError, saveProfile, sourcesDir } from "@/lib/profile-store";
import { generateThemeToml } from "@/lib/theme-toml";
import { remoteServerConfig } from "@/lib/remote-config";

// The Rust workspace this app's implementation sibling lives in — where
// the `purpose` binary is built and where .purpose/factory/registry.json
// (the shared build registry purpose-factory-ts's cli.listThemes/Registry
// read) lives. Only used in local-subprocess mode.
const IMPLEMENTATION_ROOT = join(process.cwd(), "..", "implementation");
const PURPOSE_BIN = join(
  IMPLEMENTATION_ROOT,
  "target",
  "debug",
  process.platform === "win32" ? "purpose.exe" : "purpose",
);

/**
 * Runs the build via whichever transport is configured: a remote
 * `purpose serve` server (PURPOSE_SERVE_URL + PURPOSE_SERVE_TOKEN set) if
 * present, otherwise the local `purpose` subprocess — the original,
 * same-machine behavior. Sources are already on the server in remote mode
 * (pushed at upload time by sources/upload/route.ts); in local mode a
 * theme.toml is generated pointing at the profile's local sources dir.
 */
async function runBuild(profile: Awaited<ReturnType<typeof loadProfile>>): Promise<ThemeModel> {
  const remote = remoteServerConfig();
  if (remote) {
    return http.buildTheme(
      profile.id,
      {
        urls: profile.sources.links.map((l) => l.url),
        model:
          profile.model.kind === "pretrained"
            ? { kind: "pretrained", repo: profile.model.repo, revision: profile.model.revision }
            : { kind: "scratch" },
      },
      remote,
    );
  }

  const tomlDir = await mkdtemp(join(tmpdir(), "purpose-profile-"));
  try {
    const tomlPath = join(tomlDir, "theme.toml");
    const outDir = join(PROFILES_ROOT, profile.id, "model");
    const toml = generateThemeToml(profile, sourcesDir(profile.id));
    await writeFile(tomlPath, toml);
    await mkdir(outDir, { recursive: true });

    return cli.buildTheme(tomlPath, {
      binPath: PURPOSE_BIN,
      cwd: IMPLEMENTATION_ROOT,
      out: outDir,
    });
  } finally {
    await rm(tomlDir, { recursive: true, force: true });
  }
}

export async function POST(_req: NextRequest, { params }: { params: { id: string } }) {
  let profile;
  try {
    profile = await loadProfile(params.id);
  } catch (err) {
    if (err instanceof ProfileNotFoundError) {
      return NextResponse.json({ error: err.message }, { status: 404 });
    }
    throw err;
  }

  if (profile.sources.files.length === 0 && profile.sources.links.length === 0) {
    return NextResponse.json(
      { error: "profile has no sources — add at least one file or link before training" },
      { status: 400 },
    );
  }

  profile.lastBuild = { startedAt: new Date().toISOString(), status: "running" };
  await saveProfile(profile);

  try {
    const result = await runBuild(profile);

    profile.lastBuild = {
      startedAt: profile.lastBuild.startedAt,
      finishedAt: new Date().toISOString(),
      status: "done",
      result: {
        path: result.path,
        documentCount: result.documentCount,
        exampleCount: result.exampleCount,
        vocabSize: result.vocabSize,
      },
    };
    await saveProfile(profile);
    return NextResponse.json(profile);
  } catch (err) {
    const message =
      err instanceof cli.CliBridgeError
        ? `${err.message}${err.stderr ? `\n${err.stderr}` : ""}`
        : err instanceof http.HttpClientError
          ? err.message
          : err instanceof Error
            ? err.message
            : String(err);

    profile.lastBuild = {
      startedAt: profile.lastBuild.startedAt,
      finishedAt: new Date().toISOString(),
      status: "error",
      error: message,
    };
    await saveProfile(profile);
    return NextResponse.json({ error: message, profile }, { status: 500 });
  }
}
