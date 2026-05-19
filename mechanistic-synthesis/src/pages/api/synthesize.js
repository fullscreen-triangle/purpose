import { z } from "zod";
import { getProvider } from "@/lib/llm";
import {
  SYNTHESIS_SYSTEM,
  FEDERATION_DRAFT_SYSTEM,
  INTEGRATION_SYSTEM,
  formatUserMessage,
  formatIntegrationMessage,
} from "@/lib/prompts";
import { buildPackContext, selectPacks } from "@/lib/knowledge-packs";
import {
  getFederationModels,
  getIntegrationModel,
  federationMetadata,
  floorOf,
} from "@/lib/federation";

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

const FEDERATION_ENABLED =
  (process.env.FEDERATION_ENABLED || "true").toLowerCase() !== "false";

const MAX_DRAFT_TOKENS = 4096;
const MAX_INTEGRATION_TOKENS = 8192;

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

  // Resolve knowledge packs (re-validated against current text).
  const fupText = (body.followups || [])
    .map((f) => `${f.question} ${f.answer || ""}`)
    .join("\n");
  const haystack = `${body.description}\n${fupText}`;
  const reselected = new Set(selectPacks(haystack));
  const finalPackIds = (body.packs || []).filter((id) => reselected.has(id));
  const packsToUse = finalPackIds.length > 0 ? finalPackIds : [...reselected];
  const packContext = buildPackContext(packsToUse);

  const draftSystemBase = packContext
    ? `${FEDERATION_DRAFT_SYSTEM}\n\n${packContext}`
    : FEDERATION_DRAFT_SYSTEM;
  const singleSystemBase = packContext
    ? `${SYNTHESIS_SYSTEM}\n\n${packContext}`
    : SYNTHESIS_SYSTEM;
  const integrationSystem = packContext
    ? `${INTEGRATION_SYSTEM}\n\n${packContext}`
    : INTEGRATION_SYSTEM;

  const userMessage = formatUserMessage(body.description, body.followups);

  // Set headers for chunked streaming.
  res.writeHead(200, {
    "Content-Type": "text/plain; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Transfer-Encoding": "chunked",
  });

  const writeMeta = (meta) => {
    // Emit a single-line JSON metadata sentinel that the client parses.
    // Sentinel prefix is recognised by the client and stripped from the
    // rendered Markdown body.
    res.write(`<<META>>${JSON.stringify(meta)}<<END>>\n`);
    if (typeof res.flush === "function") res.flush();
  };

  // -------- Federation disabled path: original single-model behaviour. --------
  if (!FEDERATION_ENABLED) {
    try {
      const integration = getIntegrationModel();
      writeMeta({
        phase: "single",
        draft_models: [integration],
        integration_model: null,
        aggregate_floor: floorOf(integration),
        confidence: 1 - floorOf(integration) / 100,
      });
      const stream = provider.stream({
        model: integration,
        system: singleSystemBase,
        messages: [{ role: "user", content: userMessage }],
        maxTokens: MAX_INTEGRATION_TOKENS,
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
    return;
  }

  // -------- Federation-enabled path: parallel drafts + integration. --------
  const draftModels = getFederationModels();
  writeMeta({ phase: "drafting", draft_models: draftModels });

  const draftPromises = draftModels.map(async (modelId) => {
    try {
      const text = await provider.chat({
        model: modelId,
        system: draftSystemBase,
        messages: [{ role: "user", content: userMessage }],
        maxTokens: MAX_DRAFT_TOKENS,
      });
      return { modelId, text, ok: true };
    } catch (e) {
      return { modelId, error: e.message || String(e), ok: false };
    }
  });

  const draftResults = await Promise.all(draftPromises);
  const ok = draftResults.filter((d) => d.ok);

  // If all drafts failed, surface the error.
  if (ok.length === 0) {
    if (!res.writableEnded) {
      res.write(
        `\n\n[FEDERATION FAILURE] all ${draftModels.length} draft calls failed. First error: ${draftResults[0].error}`
      );
      res.end();
    }
    return;
  }

  const meta = federationMetadata(ok.map((d) => d.modelId));
  meta.phase = "integrating";
  meta.failed_models = draftResults.filter((d) => !d.ok).map((d) => d.modelId);
  writeMeta(meta);

  // If only one draft succeeded, return it directly (no integration needed;
  // the integration call would just rephrase a single draft).
  if (ok.length === 1) {
    res.write(ok[0].text);
    res.end();
    return;
  }

  // Otherwise integrate. Stream the integration model's output.
  const integrationModel = getIntegrationModel();
  const integrationUserMessage = formatIntegrationMessage(
    body.description,
    body.followups,
    ok
  );

  try {
    const stream = provider.stream({
      model: integrationModel,
      system: integrationSystem,
      messages: [{ role: "user", content: integrationUserMessage }],
      maxTokens: MAX_INTEGRATION_TOKENS,
    });
    for await (const chunk of stream) {
      res.write(chunk);
      if (typeof res.flush === "function") res.flush();
    }
    res.end();
  } catch (e) {
    // Integration failed; fall back to the longest successful draft.
    if (!res.writableEnded) {
      const fallback = ok.reduce((a, b) =>
        (a.text || "").length >= (b.text || "").length ? a : b
      );
      res.write(`\n\n[INTEGRATION ERROR, FALLING BACK TO DRAFT ${fallback.modelId}]\n\n`);
      res.write(fallback.text);
      res.end();
    }
  }
}
