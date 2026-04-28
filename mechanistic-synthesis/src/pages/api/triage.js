import { z } from "zod";
import { getAnthropicClient, TRIAGE_MODEL } from "@/lib/anthropic";
import { TRIAGE_SYSTEM, formatUserMessage } from "@/lib/prompts";

export const config = {
  maxDuration: 60,
};

const Body = z.object({
  description: z.string().min(20, "description must be at least 20 characters"),
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

  try {
    const message = await client.messages.create({
      model: TRIAGE_MODEL,
      max_tokens: 1024,
      system: TRIAGE_SYSTEM,
      messages: [
        {
          role: "user",
          content: formatUserMessage(body.description, body.followups),
        },
      ],
    });

    const text = message.content
      .filter((b) => b.type === "text")
      .map((b) => b.text)
      .join("");

    let parsed;
    try {
      parsed = JSON.parse(extractJson(text));
    } catch (e) {
      return res
        .status(502)
        .json({ error: "triage model returned non-JSON output", raw: text });
    }

    if (!parsed || (parsed.status !== "ready" && parsed.status !== "needs_info")) {
      return res
        .status(502)
        .json({ error: "triage model returned invalid status", raw: parsed });
    }

    return res.status(200).json(parsed);
  } catch (e) {
    return res.status(500).json({ error: e.message || String(e) });
  }
}

/**
 * Extract a JSON object substring from arbitrary text. Models occasionally
 * wrap JSON in code fences or prose despite instructions.
 */
function extractJson(text) {
  const trimmed = text.trim();
  if (trimmed.startsWith("{") && trimmed.endsWith("}")) return trimmed;
  const fenceMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
  if (fenceMatch) return fenceMatch[1].trim();
  const start = trimmed.indexOf("{");
  const end = trimmed.lastIndexOf("}");
  if (start !== -1 && end !== -1 && end > start) {
    return trimmed.slice(start, end + 1);
  }
  return trimmed;
}
