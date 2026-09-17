// @buhera/purpose-factory-client — public surface.
//
// Builds and discovers theme-specific models produced by `purpose-factory`
// (the Rust crate). Training happens entirely in the Rust process; this
// package's job is driving that process and reading its output registry —
// never loading or serving a model itself. A consuming framework loads the
// exported model.safetensors/config.json/tokenizer.json bundle with
// whatever inference stack it already uses.

export type { ThemeModel, ModelConfig } from "./types.js";

export { buildTheme, listThemes, CliBridgeError } from "./cli-bridge.js";
export type { CliBridgeOptions } from "./cli-bridge.js";

export { Registry } from "./registry.js";
