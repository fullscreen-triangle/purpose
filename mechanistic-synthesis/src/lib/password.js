// Password check for the single platform user (Node runtime only).
//
// The password itself is never stored or configured anywhere: only its
// scrypt hash, produced locally by `node scripts/hash-password.mjs`, in the
// PLATFORM_PASSWORD_HASH env var. Format: scrypt$N$r$p$<salt b64>$<hash b64>.

import { scrypt, timingSafeEqual } from "node:crypto";

export function scryptAsync(password, salt, keylen, opts) {
  return new Promise((resolve, reject) =>
    scrypt(password, salt, keylen, { ...opts, maxmem: 256 * 1024 * 1024 }, (err, key) =>
      err ? reject(err) : resolve(key),
    ),
  );
}

/** Resolves true iff `password` matches `stored`; false for a malformed hash. */
export async function verifyPassword(password, stored) {
  const parts = (stored || "").split("$");
  if (parts.length !== 6 || parts[0] !== "scrypt") return false;
  const [, N, r, p, saltB64, hashB64] = parts;
  const expected = Buffer.from(hashB64, "base64");
  if (expected.length < 32) return false;
  const actual = await scryptAsync(password, Buffer.from(saltB64, "base64"), expected.length, {
    N: Number(N),
    r: Number(r),
    p: Number(p),
  });
  return timingSafeEqual(actual, expected);
}
