// Reads `.purpose/factory/registry.json` directly, without spawning the
// `purpose` binary — for a Node app that wants to enumerate available theme
// models cheaply and often (e.g. on every request) rather than shell out
// each time.

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fromRaw, type ThemeModel } from "./types.js";

export class Registry {
  private constructor(private readonly path: string) {}

  /** `root` is a project root containing `.purpose/factory/registry.json`. */
  static atRoot(root: string): Registry {
    return new Registry(join(root, ".purpose", "factory", "registry.json"));
  }

  /** `path` is the registry.json file itself. */
  static atPath(path: string): Registry {
    return new Registry(path);
  }

  async load(): Promise<ThemeModel[]> {
    let raw: string;
    try {
      raw = await readFile(this.path, "utf8");
    } catch (err) {
      if (isNoEntError(err)) {
        return [];
      }
      throw err;
    }
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.map(fromRaw) : [];
  }

  async find(name: string): Promise<ThemeModel | undefined> {
    const models = await this.load();
    return models.find((m) => m.name === name);
  }
}

function isNoEntError(err: unknown): boolean {
  return (
    typeof err === "object" &&
    err !== null &&
    "code" in err &&
    (err as { code?: unknown }).code === "ENOENT"
  );
}
