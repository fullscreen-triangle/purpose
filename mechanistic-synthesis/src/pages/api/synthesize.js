import { z } from "zod";
import { getProvider, synthesisModel } from "@/lib/llm";
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

  let provider;
  try {
    provider = getProvider();
  } catch (e) {
    return res.status(500).json({ error: e.message });
  }

  // Re-validate the requested pack ids against current text.
  const fupText = (body.followups || [])
    .map((f) => `${f.question} ${f.answer || ""}`)
    .join("\n");
  const haystack = `${body.description}\n${fupText}`;
  const reselected = new Set(selectPacks(haystack));
  const finalPackIds = (body.packs || []).filter((id) => reselected.has(id));
  const packsToUse = finalPackIds.length > 0 ? finalPackIds : [...reselected];
  const packContext = buildPackContext(packsToUse);
  const systemPrompt = packContext
    ? `${SYNTHESIS_SYSTEM}\n\n${packContext}`
    : SYNTHESIS_SYSTEM;

  res.writeHead(200, {
    "Content-Type": "text/plain; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Transfer-Encoding": "chunked",
  });

  try {
    const stream = provider.stream({
      model: synthesisModel(),
      system: systemPrompt,
      messages: [
        {
          role: "user",
          content: formatUserMessage(body.description, body.followups),
        },
      ],
      maxTokens: 8192,
    });

    for await (const chunk of stream) {
      res.write(chunk);
      if (typeof res.flush === "function") res.flush();
    }
    res.end();
  } catch (e) {
    if (!res.writableEnded) {
      res.write(`\n\n[STREAM ERROR] ${e.message || String(e)}`);
      res.end();
    }
  }
}
