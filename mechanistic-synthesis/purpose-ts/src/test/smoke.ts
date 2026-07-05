// The long-grass smoke test (buhera-specifications.md §10).
//
// Build a session with three notes — two about TUM/AIMe (overlapping the
// goal), one about milk (disjoint) — carry for goal {TUM, AIMe}, and
// assert note1+note2 are kept, note3 is dropped, floor > 0, residues present.
// Runnable directly: `node dist/test/smoke.js`.

import { Session, type Step } from "../index.js";

function mkStep(id: string, terms: string[], cost: number): Step {
  return { id, terms: new Set(terms), cost, timestamp: terms.length };
}

const s = new Session();
s.addStep(mkStep("note1", ["tum", "founded", "1868"], 24));
s.addStep(mkStep("note2", ["aime", "registry", "tum", "munich"], 30));
s.addStep(mkStep("note3", ["buy", "milk", "corner", "shop"], 28));

const c = s.carry({ goal: { terms: new Set(["tum", "aime"]) }, budget: 500 });

let failures = 0;
function check(name: string, cond: boolean): void {
  if (cond) {
    console.log(`  ok   ${name}`);
  } else {
    console.log(`  FAIL ${name}`);
    failures++;
  }
}

console.log("smoke test — buhera §10:");
check("c.ok === true", c.ok === true);
if (c.ok) {
  check("keep contains note1", c.keep.includes("note1"));
  check("keep contains note2", c.keep.includes("note2"));
  check("dropped contains note3", c.dropped.includes("note3"));
  check("ambientFloor > 0", c.ambientFloor > 0);
  check("residueMap has note1", c.residueMap.has("note1"));
  check("residueMap has note2", c.residueMap.has("note2"));
  check(
    "every residue >= floor",
    [...c.residueMap.values()].every((r) => r >= c.ambientFloor),
  );
  console.log(
    `\n  keep=${JSON.stringify(c.keep)} regenerable=${JSON.stringify(
      c.regenerable,
    )} dropped=${JSON.stringify(c.dropped)} floor=${c.ambientFloor}`,
  );
}

if (failures > 0) {
  console.error(`\nSMOKE FAILED: ${failures} check(s)`);
  process.exit(1);
} else {
  console.log("\nSMOKE PASSED");
}
