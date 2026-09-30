import { z } from "zod";
import { verifyPassword } from "@/lib/password";
import { SESSION_COOKIE, SESSION_HOURS, createSession, sessionSecret } from "@/lib/session";

const Body = z.object({ password: z.string().min(1).max(512) });

// Every failed attempt costs the caller this long, which bounds guessing
// to a few attempts per second from any one connection.
const FAILURE_DELAY_MS = 800;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "method not allowed" });
  }

  const secret = sessionSecret();
  const hash = process.env.PLATFORM_PASSWORD_HASH;
  if (!secret || !hash) {
    return res.status(503).json({
      error:
        "Login is not configured: set PLATFORM_PASSWORD_HASH and PLATFORM_SESSION_SECRET " +
        "(run `node scripts/hash-password.mjs`).",
    });
  }

  let body;
  try {
    body = Body.parse(req.body);
  } catch {
    return res.status(400).json({ error: "password required" });
  }

  if (!(await verifyPassword(body.password, hash))) {
    await sleep(FAILURE_DELAY_MS);
    return res.status(401).json({ error: "wrong password" });
  }

  const secure = process.env.NODE_ENV === "production" ? "; Secure" : "";
  res.setHeader(
    "Set-Cookie",
    `${SESSION_COOKIE}=${await createSession(secret)}; Path=/; HttpOnly; SameSite=Strict; ` +
      `Max-Age=${SESSION_HOURS * 3600}${secure}`,
  );
  return res.status(200).json({ ok: true });
}
