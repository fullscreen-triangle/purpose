// Puts the whole site behind the single-user login. Pages redirect to
// /login; API routes answer 401. Fails closed: without a configured
// PLATFORM_SESSION_SECRET no session can verify, so nothing but the login
// page (which explains what is missing) is reachable.

import { NextResponse } from "next/server";
import { SESSION_COOKIE, sessionSecret, verifySession } from "@/lib/session";

export async function middleware(req) {
  const ok = await verifySession(req.cookies.get(SESSION_COOKIE)?.value, sessionSecret());
  if (ok) return NextResponse.next();

  if (req.nextUrl.pathname.startsWith("/api/")) {
    return NextResponse.json({ error: "not logged in" }, { status: 401 });
  }
  const url = req.nextUrl.clone();
  url.pathname = "/login";
  url.search = `?next=${encodeURIComponent(req.nextUrl.pathname + req.nextUrl.search)}`;
  return NextResponse.redirect(url);
}

export const config = {
  // Everything except the login page, the auth API, and static assets.
  matcher: ["/((?!login|api/auth|_next/static|_next/image|favicon.ico|.*\\.(?:png|jpg|jpeg|svg|ico|webp|woff2?)$).*)"],
};
