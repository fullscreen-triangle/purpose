import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Purpose Profiles",
  description: "Personal knowledge profiles for purpose-factory theme models.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen font-sans antialiased">
        <nav className="border-b border-dark/10 dark:border-light/10">
          <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-3 text-sm">
            <Link href="/" className="font-semibold tracking-tight">
              Purpose
            </Link>
            <Link href="/profiles" className="opacity-70 transition hover:opacity-100">
              Profiles →
            </Link>
          </div>
        </nav>
        <div className="mx-auto max-w-3xl px-6 py-10">{children}</div>
      </body>
    </html>
  );
}
