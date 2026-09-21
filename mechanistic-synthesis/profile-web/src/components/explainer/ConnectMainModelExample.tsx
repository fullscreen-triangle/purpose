"use client";

import { useState } from "react";

interface Profile {
  id: string;
  name: string;
  domain: string;
  floor: number; // illustrative normalized floor, matching the paper's own convention
  locked: boolean;
}

const EXAMPLE_PROFILES: Profile[] = [
  { id: "p1", name: "Alex's profile", domain: "protein folding papers", floor: 0.09, locked: true },
  { id: "p2", name: "Sam's profile", domain: "distributed systems notes", floor: 0.14, locked: true },
  { id: "p3", name: "Jordan's profile", domain: "cooking recipes", floor: 0.71, locked: false },
];

const STEPS = [
  {
    title: "1. You ask a question",
    body: "“What's a robust way to replicate state across regions?” — this goes to the main model, not to any profile directly.",
  },
  {
    title: "2. The main model routes, blind to content",
    body: "The route graph has seen closure-shapes like this before pointing at Sam's and Jordan's profiles. It has never read either profile's actual notes.",
  },
  {
    title: "3. Selected profiles are queried concurrently",
    body: "Each profile runs its own elimination: it rules out candidate answers rather than asserting one, and exposes only a column — never its internal state.",
  },
  {
    title: "4. Extinction-lock check",
    body: "Sam's and Alex's answer phases become categorically indistinguishable — they lock. Jordan's profile (cooking recipes) doesn't relate to this query at all and stays distinguishable.",
  },
  {
    title: "5. Federate the locked subset only",
    body: "Jordan's profile is excluded, not averaged in with a low weight — partition extinction is binary, so there's no partial-trust option to fall back on.",
  },
  {
    title: "6. You get an honest answer",
    body: "A floor-bounded answer with its residual stated, built only from profiles that actually locked. If nothing had locked, you'd get a decline instead of a guess.",
  },
];

export function ConnectMainModelExample() {
  const [step, setStep] = useState(0);
  const current = STEPS[step] ?? STEPS[0]!;
  const showLocking = step >= 3;

  return (
    <div className="rounded-lg border border-white/10 bg-white/5 p-5">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-semibold text-light">{current.title}</h4>
        <div className="flex gap-2">
          <button
            onClick={() => setStep((s) => Math.max(0, s - 1))}
            disabled={step === 0}
            className="rounded border border-white/20 px-2 py-1 text-xs text-light/80 disabled:opacity-30"
          >
            Back
          </button>
          <button
            onClick={() => setStep((s) => Math.min(STEPS.length - 1, s + 1))}
            disabled={step === STEPS.length - 1}
            className="rounded bg-primary px-2 py-1 text-xs text-white disabled:opacity-30"
          >
            Next
          </button>
        </div>
      </div>
      <p className="mt-1 text-xs text-light/60">{current.body}</p>

      <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-3">
        {EXAMPLE_PROFILES.map((p) => {
          const locked = showLocking && p.locked;
          const dimmed = showLocking && !p.locked;
          return (
            <div
              key={p.id}
              className={`rounded border p-3 text-xs transition ${
                locked
                  ? "border-primaryDark bg-primaryDark/10"
                  : dimmed
                    ? "border-white/10 bg-white/0 opacity-40"
                    : "border-white/15 bg-white/5"
              }`}
            >
              <p className="font-semibold text-light">{p.name}</p>
              <p className="mt-0.5 text-light/50">{p.domain}</p>
              <p className="mt-1 text-light/40">floor {p.floor.toFixed(2)}</p>
              {showLocking && (
                <p className={`mt-1 font-medium ${locked ? "text-primaryDark" : "text-light/30"}`}>
                  {locked ? "extinction-locked" : "excluded"}
                </p>
              )}
            </div>
          );
        })}
      </div>

      {step === STEPS.length - 1 && (
        <div className="mt-4 rounded border border-primaryDark/40 bg-primaryDark/10 p-3 text-xs text-light/80">
          Answer certified from Alex&rsquo;s and Sam&rsquo;s profiles, floor &le; 0.14 + residual.
          Jordan&rsquo;s profile never contributed and never had its content read by the main model.
        </div>
      )}

      <div className="mt-4 flex gap-1">
        {STEPS.map((_, i) => (
          <div key={i} className={`h-1 flex-1 rounded ${i <= step ? "bg-primaryDark" : "bg-white/10"}`} />
        ))}
      </div>
    </div>
  );
}
