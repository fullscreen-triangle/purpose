/**
 * Lightweight localStorage helpers for synthesis history.
 * History is keyed by a UUID-like id and stored under "ms-history".
 * No auth, no backend — entirely client-side.
 */

const HISTORY_KEY = "ms-history-v1";

function uid() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function read() {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function write(items) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(HISTORY_KEY, JSON.stringify(items));
  } catch {
    // localStorage may be full or disabled; fail silently
  }
}

export function listHistory() {
  return read().sort((a, b) => b.createdAt - a.createdAt);
}

export function getHistoryItem(id) {
  return read().find((it) => it.id === id) || null;
}

export function saveHistoryItem({ description, followups, synthesis, summary, field }) {
  const item = {
    id: uid(),
    createdAt: Date.now(),
    description,
    followups: followups || [],
    synthesis,
    summary: summary || "",
    field: field || "",
  };
  const items = read();
  items.push(item);
  // Keep at most 50 items (FIFO eviction)
  while (items.length > 50) items.shift();
  write(items);
  return item;
}

export function deleteHistoryItem(id) {
  const items = read().filter((it) => it.id !== id);
  write(items);
}

export function clearHistory() {
  write([]);
}
