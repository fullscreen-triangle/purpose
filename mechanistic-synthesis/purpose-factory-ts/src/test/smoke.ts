// Smoke test for @buhera/purpose-factory-client's Registry reader.
// Exercises the no-subprocess path only (cli-bridge.ts needs the `purpose`
// binary on PATH, which isn't guaranteed in every environment this runs in).
// Runnable directly: `node dist/test/smoke.js`.

import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Registry } from "../index.js";

let failures = 0;
function check(name: string, cond: boolean): void {
  if (cond) {
    console.log(`  ok   ${name}`);
  } else {
    console.log(`  FAIL ${name}`);
    failures++;
  }
}

async function main(): Promise<void> {
  console.log("smoke test — purpose-factory-client Registry:");

  const dir = await mkdtemp(join(tmpdir(), "purpose-factory-ts-"));
  const registryPath = join(dir, ".purpose", "factory", "registry.json");

  try {
    const empty = await Registry.atPath(registryPath).load();
    check("missing registry loads as empty array", empty.length === 0);

    const { mkdir } = await import("node:fs/promises");
    await mkdir(join(dir, ".purpose", "factory"), { recursive: true });
    await writeFile(
      registryPath,
      JSON.stringify([
        {
          name: "accountable-compilation",
          path: join(dir, "out"),
          document_count: 1,
          example_count: 294,
          vocab_size: 1024,
        },
      ]),
    );

    const models = await Registry.atPath(registryPath).load();
    check("loaded one model", models.length === 1);
    check("name preserved", models[0]?.name === "accountable-compilation");
    check(
      "snake_case fields converted to camelCase",
      models[0]?.documentCount === 1 &&
        models[0]?.exampleCount === 294 &&
        models[0]?.vocabSize === 1024,
    );

    const found = await Registry.atPath(registryPath).find(
      "accountable-compilation",
    );
    check("find() locates by name", found !== undefined);

    const notFound = await Registry.atPath(registryPath).find("nonexistent");
    check("find() returns undefined for unknown name", notFound === undefined);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }

  if (failures > 0) {
    console.error(`\nSMOKE FAILED: ${failures} check(s)`);
    process.exit(1);
  } else {
    console.log("\nSMOKE PASSED");
  }
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});
