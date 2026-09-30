// Single-user session cookie, signed with HMAC-SHA256 over its expiry.
//
// Uses only Web Crypto, so the same code runs in the edge middleware (which
// checks every request) and in the Node API route (which issues the cookie).
// The cookie holds no identity — there is one user — only "valid until".

export const SESSION_COOKIE = "ms_session";
export const SESSION_HOURS = 12;

const enc = new TextEncoder();

/** The signing secret, or null when it is missing or too short to be safe. */
export function sessionSecret() {
  const s = process.env.PLATFORM_SESSION_SECRET;
  return s && s.length >= 32 ? s : null;
}

function b64url(bytes) {
  let bin = "";
  for (const b of new Uint8Array(bytes)) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function hmac(secret, message) {
  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  return b64url(await crypto.subtle.sign("HMAC", key, enc.encode(message)));
}

/** Equal-time string comparison, so a forged signature cannot be found byte by byte. */
function timingSafeEqualStr(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

/** A cookie value valid for SESSION_HOURS: "<expiry-ms>.<signature>". */
export async function createSession(secret) {
  const exp = Date.now() + SESSION_HOURS * 3600 * 1000;
  return `${exp}.${await hmac(secret, `session:${exp}`)}`;
}

/** True only for an unexpired value signed with `secret`. */
export async function verifySession(value, secret) {
  if (!value || !secret) return false;
  const dot = value.indexOf(".");
  if (dot < 1) return false;
  const exp = value.slice(0, dot);
  if (!/^\d+$/.test(exp) || Number(exp) < Date.now()) return false;
  return timingSafeEqualStr(value.slice(dot + 1), await hmac(secret, `session:${exp}`));
}
