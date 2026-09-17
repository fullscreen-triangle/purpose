// Mirrors purpose-factory's Rust types (crates/purpose-factory/src/factory.rs,
// theme_config.rs). Kept in sync by hand — the Rust side is the source of
// truth; this is the JSON shape it serializes.

/** A built theme model, as recorded in `.purpose/factory/registry.json`. */
export interface ThemeModel {
  name: string;
  /** Directory containing model.safetensors, config.json, tokenizer.json. */
  path: string;
  documentCount: number;
  exampleCount: number;
  vocabSize: number;
}

/** Raw registry JSON shape (Rust's serde field names, snake_case). */
interface ThemeModelRaw {
  name: string;
  path: string;
  document_count: number;
  example_count: number;
  vocab_size: number;
}

export function fromRaw(raw: ThemeModelRaw): ThemeModel {
  return {
    name: raw.name,
    path: raw.path,
    documentCount: raw.document_count,
    exampleCount: raw.example_count,
    vocabSize: raw.vocab_size,
  };
}

/** The exported model config, `config.json` inside a theme model's directory. */
export interface ModelConfig {
  architecture: string;
  vocabSize: number;
  nLayer: number;
  nHead: number;
  nEmbd: number;
  blockSize: number;
}

interface ModelConfigRaw {
  architecture: string;
  vocab_size: number;
  n_layer: number;
  n_head: number;
  n_embd: number;
  block_size: number;
}

export function modelConfigFromRaw(raw: ModelConfigRaw): ModelConfig {
  return {
    architecture: raw.architecture,
    vocabSize: raw.vocab_size,
    nLayer: raw.n_layer,
    nHead: raw.n_head,
    nEmbd: raw.n_embd,
    blockSize: raw.block_size,
  };
}
