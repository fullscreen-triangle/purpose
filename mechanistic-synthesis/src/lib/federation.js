/**
 * Federation policies for the FKAC pipeline.
 *
 * A federation is a set of receiver models that draft the synthesis in
 * parallel. Their outputs feed into an integration model that produces the
 * final document. Per-model floor estimates feed the aggregate floor
 * computation used for the confidence badge.
 *
 * Models can be overridden via env vars:
 *   FEDERATION_MODELS       comma-separated HF model ids for parallel drafts
 *   INTEGRATION_MODEL       HF model id for the integration call
 *
 * Floor estimates are coarse priors based on parameter count and benchmark
 * reputation, on the canonical [0, 100] S-scale where 0 = perfect, 100 =
 * worst-case alignment. They are NOT learned; they are a starting point.
 */

// Defaults are ungated, broadly-available chat models on the HF Inference
// Providers router. Override via FEDERATION_MODELS / INTEGRATION_MODEL.
const DEFAULT_FEDERATION_MODELS = [
  "Qwen/Qwen2.5-7B-Instruct",
  "microsoft/Phi-3.5-mini-instruct",
  "HuggingFaceH4/zephyr-7b-beta",
];

const DEFAULT_INTEGRATION_MODEL = "Qwen/Qwen2.5-72B-Instruct";

// Floor priors on [0, 100]. Low = stronger model.
const FLOOR_PRIORS = {
  "Qwen/Qwen2.5-7B-Instruct": 31,
  "microsoft/Phi-3.5-mini-instruct": 38,
  "HuggingFaceH4/zephyr-7b-beta": 42,
  "Qwen/Qwen2.5-72B-Instruct": 19,
  "meta-llama/Llama-3.1-8B-Instruct": 32,
  "meta-llama/Llama-3.3-70B-Instruct": 18,
  "mistralai/Mistral-7B-Instruct-v0.3": 35,
  "mistralai/Mixtral-8x7B-Instruct-v0.1": 24,
};

const SIGMA = 100;

export function getFederationModels() {
  const env = process.env.FEDERATION_MODELS;
  if (env) {
    return env.split(",").map((s) => s.trim()).filter(Boolean);
  }
  return DEFAULT_FEDERATION_MODELS;
}

export function getIntegrationModel() {
  return process.env.INTEGRATION_MODEL || DEFAULT_INTEGRATION_MODEL;
}

export function floorOf(modelId) {
  if (FLOOR_PRIORS[modelId] != null) return FLOOR_PRIORS[modelId];
  if (/70B|72B|8x22B/i.test(modelId)) return 20;
  if (/8B|7B/i.test(modelId)) return 32;
  if (/3B|mini/i.test(modelId)) return 40;
  return 30;
}

/**
 * Aggregate floor under the parallel composition (Theorem 5.3 of the FKAC
 * paper). Each receiver fails independently with probability beta_i / Sigma;
 * the joint receiver fails when all fail, i.e. probability prod(beta_i/Sigma).
 * Multiplying by Sigma gives the joint floor on the [0, Sigma] scale.
 */
export function aggregateFloor(modelIds) {
  if (!modelIds || modelIds.length === 0) return SIGMA;
  const product = modelIds.reduce((acc, id) => acc * (floorOf(id) / SIGMA), 1);
  return SIGMA * product;
}

/**
 * Confidence is a user-facing rendering: 1 - floor/Sigma. Returns a number
 * in [0, 1] suitable for display as a percentage.
 */
export function confidenceFromFloor(floor) {
  return Math.max(0, Math.min(1, 1 - floor / SIGMA));
}

export function federationMetadata(activeDraftIds) {
  const integration = getIntegrationModel();
  const succeededFloors = activeDraftIds.map(floorOf);
  const agg = aggregateFloor(activeDraftIds);
  return {
    draft_models: activeDraftIds,
    integration_model: integration,
    per_model_floors: succeededFloors,
    aggregate_floor: agg,
    confidence: confidenceFromFloor(agg),
  };
}
