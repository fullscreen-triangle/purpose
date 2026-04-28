import { z } from "zod";
import { getAnthropicClient, SYNTHESIS_MODEL } from "@/lib/anthropic";
import { SYNTHESIS_SYSTEM, formatUserMessage } from "@/lib/prompts";

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

  try {
    const stream = client.messages.stream({
      model: SYNTHESIS_MODEL,
      max_tokens: 8192,
      system: SYNTHESIS_SYSTEM,
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
