/**
 * LLM provider abstraction.
 *
 * Two concrete providers are supported:
 *   - HuggingFace Inference API (default; broad model choice, no Anthropic dependency)
 *   - Anthropic (kept as a fallback if ANTHROPIC_API_KEY is the only key present)
 *
 * Selection rule:
 *   1. If LLM_PROVIDER is explicitly set ("huggingface" or "anthropic"), use it.
 *   2. Else if HUGGINGFACE_API_KEY is present, use HuggingFace.
 *   3. Else if ANTHROPIC_API_KEY is present, use Anthropic.
 *   4. Else throw with a clear error.
 *
 * The provider interface is intentionally narrow:
 *   - chat({ system, messages, model, maxTokens, responseFormat }) -> string
 *   - stream({ system, messages, model, maxTokens }) -> async iterable of text chunks
 *
 * Adding a third provider (Together AI, Cohere, local Ollama) is one class.
 */

import Anthropic from "@anthropic-ai/sdk";

// ============================================================================
// Provider selection
// ============================================================================

let _provider = null;

export function getProvider() {
  if (_provider) return _provider;

  const explicit = (process.env.LLM_PROVIDER || "").toLowerCase();
  if (explicit === "huggingface") {
    _provider = new HuggingFaceProvider();
  } else if (explicit === "anthropic") {
    _provider = new AnthropicProvider();
  } else if (process.env.HUGGINGFACE_API_KEY) {
    _provider = new HuggingFaceProvider();
  } else if (process.env.ANTHROPIC_API_KEY) {
    _provider = new AnthropicProvider();
  } else {
    throw new Error(
      "No LLM provider configured. Set HUGGINGFACE_API_KEY (recommended) " +
        "or ANTHROPIC_API_KEY in .env.local. See .env.local.example."
    );
  }
  return _provider;
}

// ============================================================================
// Model resolution
// ============================================================================

export function triageModel() {
  const p = (process.env.LLM_PROVIDER || "").toLowerCase();
  if (p === "anthropic" || (!process.env.HUGGINGFACE_API_KEY && process.env.ANTHROPIC_API_KEY)) {
    return process.env.ANTHROPIC_TRIAGE_MODEL || "claude-sonnet-4-6";
  }
  // Defaults to a widely-available, ungated 7B chat model.
  return process.env.HUGGINGFACE_TRIAGE_MODEL || "Qwen/Qwen2.5-7B-Instruct";
}

export function synthesisModel() {
  const p = (process.env.LLM_PROVIDER || "").toLowerCase();
  if (p === "anthropic" || (!process.env.HUGGINGFACE_API_KEY && process.env.ANTHROPIC_API_KEY)) {
    return process.env.ANTHROPIC_SYNTHESIS_MODEL || "claude-opus-4-7";
  }
  return process.env.HUGGINGFACE_SYNTHESIS_MODEL || "Qwen/Qwen2.5-72B-Instruct";
}

// ============================================================================
// HuggingFace provider (default)
// ============================================================================

class HuggingFaceProvider {
  constructor() {
    this.apiKey = process.env.HUGGINGFACE_API_KEY;
    if (!this.apiKey) {
      throw new Error(
        "HUGGINGFACE_API_KEY is not set. Get one at https://huggingface.co/settings/tokens (Read token is sufficient)."
      );
    }
    // Unified Inference Providers router. The model is sent in the request
    // body; the router selects an available provider for that model.
    // For dedicated endpoints set HUGGINGFACE_BASE_URL to override.
    this.baseUrl =
      process.env.HUGGINGFACE_BASE_URL ||
      "https://router.huggingface.co/v1/chat/completions";
  }

  endpoint() {
    return this.baseUrl;
  }

  buildBody({ system, messages, model, maxTokens, stream }) {
    const all = [];
    if (system) all.push({ role: "system", content: system });
    for (const m of messages) all.push(m);
    return {
      model,
      messages: all,
      max_tokens: maxTokens || 1024,
      temperature: 0.4,
      stream: !!stream,
    };
  }

  async chat({ system, messages, model, maxTokens }) {
    const res = await fetch(this.endpoint(), {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(
        this.buildBody({ system, messages, model, maxTokens, stream: false })
      ),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(
        `HuggingFace API ${res.status} for model "${model}". ` +
          `If 401/403: visit the model's HF page and accept its license. ` +
          `If 404: model not available on the Inference Providers router; ` +
          `try Qwen/Qwen2.5-7B-Instruct or microsoft/Phi-3.5-mini-instruct. ` +
          `Body: ${text.slice(0, 300)}`
      );
    }
    const json = await res.json();
    const content = json?.choices?.[0]?.message?.content;
    if (typeof content !== "string") {
      throw new Error("HuggingFace response missing choices[0].message.content");
    }
    return content;
  }

  async *stream({ system, messages, model, maxTokens }) {
    const res = await fetch(this.endpoint(), {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(
        this.buildBody({ system, messages, model, maxTokens, stream: true })
      ),
    });
    if (!res.ok || !res.body) {
      const text = await res.text().catch(() => "");
      throw new Error(
        `HuggingFace streaming ${res.status} for model "${model}". ` +
          `If 401/403: accept the model's license on its HF page. ` +
          `If 404: model not available on the Inference Providers router. ` +
          `Body: ${text.slice(0, 300)}`
      );
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // SSE frames are separated by double newlines; each event is one or
      // more lines starting with "data:".
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() || "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice(5).trim();
        if (payload === "[DONE]") return;
        try {
          const obj = JSON.parse(payload);
          const delta = obj?.choices?.[0]?.delta?.content;
          if (typeof delta === "string" && delta.length > 0) {
            yield delta;
          }
        } catch {
          // Some lines are heartbeats or non-JSON; ignore.
        }
      }
    }
  }
}

// ============================================================================
// Anthropic provider (fallback when ANTHROPIC_API_KEY is configured)
// ============================================================================

class AnthropicProvider {
  constructor() {
    this.apiKey = process.env.ANTHROPIC_API_KEY;
    if (!this.apiKey) {
      throw new Error("ANTHROPIC_API_KEY is not set.");
    }
    this.client = new Anthropic({ apiKey: this.apiKey });
  }

  async chat({ system, messages, model, maxTokens }) {
    const message = await this.client.messages.create({
      model,
      max_tokens: maxTokens || 1024,
      system,
      messages: messages.filter((m) => m.role !== "system"),
    });
    return (message.content || [])
      .filter((b) => b.type === "text")
      .map((b) => b.text)
      .join("");
  }

  async *stream({ system, messages, model, maxTokens }) {
    const stream = this.client.messages.stream({
      model,
      max_tokens: maxTokens || 8192,
      system,
      messages: messages.filter((m) => m.role !== "system"),
    });
    for await (const event of stream) {
      if (
        event.type === "content_block_delta" &&
        event.delta &&
        event.delta.type === "text_delta" &&
        typeof event.delta.text === "string"
      ) {
        yield event.delta.text;
      }
    }
  }
}
