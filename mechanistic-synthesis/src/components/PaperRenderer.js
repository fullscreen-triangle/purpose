import React, { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSlug from "rehype-slug";
import rehypeAutolinkHeadings from "rehype-autolink-headings";

/**
 * Render a streamed Markdown synthesis as a paper-shaped document.
 * Includes a sticky table of contents derived from h2 headings.
 */
export default function PaperRenderer({ markdown, streaming = false }) {
  const sections = useMemo(() => extractSections(markdown), [markdown]);

  return (
    <div className="w-full max-w-6xl mx-auto grid grid-cols-12 gap-8">
      <aside className="col-span-3 lg:col-span-12 lg:order-2">
        <div className="sticky top-24 lg:static">
          <p className="text-xs uppercase tracking-wider text-dark/50 dark:text-light/50 mb-3">
            Contents
          </p>
          <nav className="space-y-1.5">
            {sections.length === 0 && streaming && (
              <p className="text-sm text-dark/40 dark:text-light/40 italic">
                generating…
              </p>
            )}
            {sections.map((s) => (
              <a
                key={s.slug}
                href={`#${s.slug}`}
                className="block text-sm text-dark/70 dark:text-light/70
                           hover:text-primary dark:hover:text-primaryDark transition"
              >
                {s.title}
              </a>
            ))}
          </nav>
        </div>
      </aside>

      <article className="col-span-9 lg:col-span-12 lg:order-1 paper-content">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[
            rehypeSlug,
            [rehypeAutolinkHeadings, { behavior: "wrap" }],
          ]}
        >
          {markdown || ""}
        </ReactMarkdown>
        {streaming && (
          <span
            className="inline-block w-2 h-5 -mb-1 bg-primary dark:bg-primaryDark
                       animate-pulse ml-0.5"
            aria-hidden
          />
        )}
      </article>
    </div>
  );
}

function extractSections(md) {
  if (!md) return [];
  const out = [];
  const lines = md.split("\n");
  for (const line of lines) {
    const m = line.match(/^##\s+(.+?)\s*$/);
    if (m) {
      const title = m[1].trim();
      out.push({ title, slug: slugify(title) });
    }
  }
  return out;
}

function slugify(s) {
  return s
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/--+/g, "-")
    .replace(/^-+|-+$/g, "");
}
