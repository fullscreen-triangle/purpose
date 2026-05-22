/**
 * Knowledge-pack registry and selector.
 *
 * A knowledge pack is a self-contained directory of Markdown reference
 * material that the synthesis stage can include in its system prompt
 * when the user's description matches the pack's trigger keywords.
 *
 * Layout on disk:
 *   knowledge-packs/<pack-id>/
 *     manifest.json
 *     <one or more .md files referenced from manifest>
 *
 * This module loads manifests lazily on first use and caches them in
 * memory for the lifetime of the server process.
 */

import fs from "fs";
import path from "path";

const ROOT = path.join(process.cwd(), "knowledge-packs");

let _registry = null;

function loadRegistry() {
  if (_registry) return _registry;
  _registry = [];
  if (!fs.existsSync(ROOT)) return _registry;

  const entries = fs.readdirSync(ROOT, { withFileTypes: true });
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const manifestPath = path.join(ROOT, entry.name, "manifest.json");
    if (!fs.existsSync(manifestPath)) continue;
    try {
      const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
      manifest._dir = path.join(ROOT, entry.name);
      _registry.push(manifest);
    } catch (e) {
      console.warn(`failed to load knowledge pack at ${manifestPath}: ${e.message}`);
    }
  }
  return _registry;
}

/**
 * Public: list all packs (manifest objects, without their file contents).
 */
export function listPacks() {
  return loadRegistry().map((m) => ({
    id: m.id,
    name: m.name,
    short_label: m.short_label,
    summary: m.summary,
    source_repo: m.source_repo,
  }));
}

/**
 * Public: given the user's description (and optional summary returned by
 * triage), return an array of pack ids that should be activated.
 *
 * Matching is keyword-based: case-insensitive substring match against the
 * concatenated text. The pack's `trigger_match` field controls semantics:
 *   - "any" (default): match if at least one keyword is present
 *   - "all": match if every keyword is present
 *
 * Token budget is enforced across all activated packs: packs are added in
 * registry order until the budget would be exceeded; remaining packs are
 * skipped.
 */
export function selectPacks(haystack, { budget = 60000 } = {}) {
  const text = (haystack || "").toLowerCase();
  const registry = loadRegistry();
  const activated = [];
  let usedBudget = 0;

  for (const pack of registry) {
    const matchMode = pack.trigger_match || "any";
    const keywords = (pack.trigger_keywords || []).map((k) => k.toLowerCase());
    if (keywords.length === 0) continue;
    const matched =
      matchMode === "all"
        ? keywords.every((k) => text.includes(k))
        : keywords.some((k) => text.includes(k));
    if (!matched) continue;

    const packBudget = pack.context_budget_tokens || 8000;
    if (usedBudget + packBudget > budget) continue;
    usedBudget += packBudget;
    activated.push(pack.id);
  }
  return activated;
}

/**
 * Public: given a pack id, return concatenated Markdown content from all
 * files listed in its manifest, or `null` if the pack does not exist.
 */
export function loadPackContent(packId) {
  const registry = loadRegistry();
  const pack = registry.find((p) => p.id === packId);
  if (!pack) return null;

  const sections = [];
  sections.push(`## Pack: ${pack.name}`);
  if (pack.summary) sections.push(pack.summary);

  for (const fileEntry of pack.files || []) {
    const filePath = path.join(pack._dir, fileEntry.path);
    if (!fs.existsSync(filePath)) {
      sections.push(`<!-- missing file: ${fileEntry.path} -->`);
      continue;
    }
    const content = fs.readFileSync(filePath, "utf8");
    sections.push(`### ${fileEntry.description || fileEntry.path}\n\n${content}`);
  }
  return sections.join("\n\n");
}

/**
 * Public: given an array of pack ids, return a single string of
 * Markdown-formatted reference material suitable for injection into the
 * synthesis system prompt.
 */
export function buildPackContext(packIds) {
  if (!packIds || packIds.length === 0) return "";
  const blocks = [];
  for (const id of packIds) {
    const content = loadPackContent(id);
    if (content) blocks.push(content);
  }
  if (blocks.length === 0) return "";
  return [
    "# Domain Reference Material",
    "",
    "The following authoritative reference material is provided as context. Treat it as the canonical statement of the framework used in this domain. When relevant to the researcher's experiment, integrate its concepts, validated numbers, and terminology into the synthesis. Cite specific results from this material with attribution to the pack author where appropriate.",
    "",
    ...blocks,
  ].join("\n");
}

/**
 * Public: given a pack id, return the manifest's `short_label` for UI display.
 */
export function getPackLabel(packId) {
  const registry = loadRegistry();
  const pack = registry.find((p) => p.id === packId);
  return pack ? pack.short_label || pack.name : packId;
}
