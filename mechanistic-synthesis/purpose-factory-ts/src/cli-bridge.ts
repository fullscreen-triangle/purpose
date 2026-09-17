// Spawns the `purpose` binary and parses its `--raw` JSON output, per
// integration.md §5 ("Embed Purpose in a non-Rust host: Spawn the CLI;
// communicate via `--raw` JSON on stdout"). No training happens here — this
// module only drives the Rust factory as a subprocess.

import { spawn } from "node:child_process";
import { fromRaw, type ThemeModel } from "./types.js";

export interface CliBridgeOptions {
  /** Path to the `purpose` binary. Defaults to "purpose" (must be on PATH). */
  binPath?: string;
  /** Working directory for the subprocess (affects `.purpose/` root detection). */
  cwd?: string;
}

export class CliBridgeError extends Error {
  constructor(
    message: string,
    public readonly exitCode: number | null,
    public readonly stderr: string,
  ) {
    super(message);
    this.name = "CliBridgeError";
  }
}

function run(args: string[], opts: CliBridgeOptions): Promise<string> {
  return new Promise((resolve, reject) => {
    const bin = opts.binPath ?? "purpose";
    const child = spawn(bin, args, { cwd: opts.cwd });

    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf8");
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8");
    });

    child.on("error", (err) => {
      reject(
        new CliBridgeError(
          `failed to spawn '${bin}': ${err.message}`,
          null,
          stderr,
        ),
      );
    });

    child.on("close", (code) => {
      if (code !== 0) {
        reject(
          new CliBridgeError(
            `'${bin} ${args.join(" ")}' exited with code ${code}`,
            code,
            stderr,
          ),
        );
        return;
      }
      resolve(stdout);
    });
  });
}

/**
 * Runs `purpose factory build <config> --raw [--out <out>]` and returns the
 * resulting ThemeModel. Training happens inside the Rust process; this call
 * resolves once it has finished (or rejects with the subprocess's stderr).
 */
export async function buildTheme(
  configPath: string,
  options: CliBridgeOptions & { out?: string } = {},
): Promise<ThemeModel> {
  const args = ["factory", "build", configPath, "--raw"];
  if (options.out) {
    args.push("--out", options.out);
  }
  const stdout = await run(args, options);
  return fromRaw(JSON.parse(stdout));
}

/**
 * Runs `purpose factory list --raw` and returns the recorded theme models.
 * Prefer `Registry.load` (registry.ts) when you already know the registry
 * file path — it reads the JSON directly without spawning a process.
 */
export async function listThemes(
  options: CliBridgeOptions & { registry?: string } = {},
): Promise<ThemeModel[]> {
  const args = ["factory", "list", "--raw"];
  if (options.registry) {
    args.push("--registry", options.registry);
  }
  const stdout = await run(args, options);
  const raw: unknown = JSON.parse(stdout);
  return Array.isArray(raw) ? raw.map(fromRaw) : [];
}
