import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Purpose Profiles",
  description: "Personal knowledge profiles for purpose-factory theme models.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen font-sans antialiased">
        <div className="mx-auto max-w-3xl px-6 py-10">{children}</div>
      </body>
    </html>
  );
}
