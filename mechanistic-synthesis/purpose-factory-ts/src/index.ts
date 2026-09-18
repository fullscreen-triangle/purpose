// @buhera/purpose-factory-client — public surface.
//
// Builds and discovers theme-specific models produced by `purpose-factory`
// (the Rust crate). Training happens entirely in the Rust process; this
// package's job is driving that process and reading its output registry —
// never loading or serving a model itself. A consuming framework loads the
// exported model.safetensors/config.json/tokenizer.json bundle with
// whatever inference stack it already uses.
//
// Two transports, same shape (buildTheme/listThemes), namespaced to avoid a
// naming collision and to make the call site's transport explicit:
//   - `cli.*` spawns the `purpose` binary as a local subprocess
//     (cli-bridge.ts) — caller and purpose-factory share a filesystem.
//   - `http.*` talks to a `purpose serve` server over the network
//     (http-client.ts), bearer-token authenticated — no shared filesystem
//     required; also the only transport with uploadSources/downloadModelFile,
//     since a local subprocess needs no upload/download step.

export type { ThemeModel, ModelConfig } from "./types.js";

export { Registry } from "./registry.js";

export * as cli from "./cli-bridge.js";
export * as http from "./http-client.js";
