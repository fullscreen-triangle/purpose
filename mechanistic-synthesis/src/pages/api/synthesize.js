import { z } from "zod";
import { getAnthropicClient, SYNTHESIS_MODEL } from "@/lib/anthropic";
import { SYNTHESIS_SYSTEM, formatUserMessage } from "@/lib/prompts";
import { buildPackContext, selectPacks } from "@/lib/knowledge-packs";

export const config = {
  maxDuration: 300,
};

const Body = z.object({
  description: z.string().min(20),
  followups: z
    .array(
      z.object({
        question: z.string(),
        answer: z.string().optional().default(""),
      })
    )
    .optional()
    .default([]),
  packs: z.array(z.string()).optional().default([]),
});

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "Method not allowed" });
  }

  let body;
  try {
    body = Body.parse(req.body);
  } catch (e) {
    return res.status(400).json({ error: e.errors?.[0]?.message || "invalid body" });
  }

  let client;
  try {
    client = getAnthropicClient();
  } catch (e) {
    return res.status(500).json({ error: e.message });
  }

  res.writeHead(200, {
    "Content-Type": "text/plain; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Transfer-Encoding": "chunked",
  });

  // Resolve the set of knowledge packs to include. The client passes the
  // pack ids the triage step returned; we re-validate here against the
  // current description + followups so a stale or tampered list is
  // automatically narrowed to packs that still match.
  const fupText = (body.followups || [])
    .map((f) => `${f.question} ${f.answer || ""}`)
    .join("\n");
  const haystack = `${body.description}\n${fupText}`;
  const reselected = new Set(selectPacks(haystack));
  const finalPackIds = (body.packs || []).filter((id) => reselected.has(id));
  // If the client did not pass any pack hints, fall back to fresh selection.
  const packsToUse =
    finalPackIds.length > 0 ? finalPackIds : [...reselected];
  const packContext = buildPackContext(packsToUse);
  const systemPrompt = packContext
    ? `${SYNTHESIS_SYSTEM}\n\n${packContext}`
    : SYNTHESIS_SYSTEM;

  try {
    const stream = client.messages.stream({
      model: SYNTHESIS_MODEL,
      max_tokens: 8192,
      system: systemPrompt,
      messages: [
        {
          role: "user",
          content: formatUserMessage(body.description, body.followups),
        },
      ],
    });

    for await (const event of stream) {
      if (
        event.type === "content_block_delta" &&
        event.delta &&
        event.delta.type === "text_delta" &&
        typeof event.delta.text === "string"
      ) {
        res.write(event.delta.text);
        if (typeof res.flush === "function") res.flush();
      }
    }
    res.end();
  } catch (e) {
    if (!res.writableEnded) {
      res.write(`\n\n[STREAM ERROR] ${e.message || String(e)}`);
      res.end();
    }
  }
}
