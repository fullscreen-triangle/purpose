// Generates a theme.toml matching purpose-factory's ThemeConfig shape
// (mechanistic-synthesis/implementation/crates/purpose-factory/src/theme_config.rs)
// from a Profile. A profile has no Rust-side concept — this is the
// translation layer.

import type { Profile } from "./types.js";

function tomlString(value: string): string {
  // TOML basic strings: backslash and quote need escaping. Using a basic
  // string (not a literal string) because Windows paths are common here and
  // the escaping is simple and well-defined either way — this keeps one
  // code path for both plain text and paths.
  const escaped = value.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
  return `"${escaped}"`;
}

function tomlStringArray(values: string[]): string {
  if (values.length === 0) return "[]";
  return `[${values.map(tomlString).join(", ")}]`;
}

export function generateThemeToml(profile: Profile, sourcesRoot: string): string {
  const lines: string[] = [];

  lines.push(`name = ${tomlString(profile.id)}`);
  lines.push("");

  lines.push("[sources]");
  if (profile.sources.files.length > 0) {
    lines.push(
      `local_files = [{ root = ${tomlString(sourcesRoot)} }]`,
    );
  } else {
    lines.push("local_files = []");
  }
  lines.push(`urls = ${tomlStringArray(profile.sources.links.map((l) => l.url))}`);
  lines.push("");

  lines.push("[model]");
  if (profile.model.kind === "pretrained") {
    lines.push("[model.pretrained]");
    lines.push(`repo = ${tomlString(profile.model.repo)}`);
    if (profile.model.revision) {
      lines.push(`revision = ${tomlString(profile.model.revision)}`);
    }
  }
  // kind === "scratch": omit [model] body entirely, letting the Rust side's
  // ScratchConfig defaults apply (theme_config.rs's default_* fns).

  return lines.join("\n") + "\n";
}
