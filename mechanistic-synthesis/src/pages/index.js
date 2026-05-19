import Head from "next/head";
import { useState, useCallback } from "react";
import { motion } from "framer-motion";

import ExperimentInput from "@/components/ExperimentInput";
import FollowupPanel from "@/components/FollowupPanel";
import PaperRenderer from "@/components/PaperRenderer";
import LoadingState from "@/components/LoadingState";
import PackBadge from "@/components/PackBadge";
import ExampleQueries from "@/components/ExampleQueries";
import FederationStatus from "@/components/FederationStatus";
import { saveHistoryItem } from "@/lib/storage";

// Recognise the streaming metadata sentinel emitted by /api/synthesize.
// Server emits: <<META>>{"phase":"drafting",...}<<END>>\n
const META_RE = /<<META>>(\{[\s\S]*?\})<<END>>\n?/g;
function extractMetaAndStrip(text) {
  let m;
  const metas = [];
  let stripped = text;
  while ((m = META_RE.exec(text)) !== null) {
    try {
      metas.push(JSON.parse(m[1]));
    } catch {
      // ignore malformed metadata
    }
  }
  stripped = text.replace(META_RE, "");
  return { metas, stripped };
}

const MAX_FOLLOWUP_ROUNDS = 3;

export default function Home() {
  const [phase, setPhase] = useState("input");
  // "input" | "triaging" | "followup" | "synthesizing" | "result" | "error"

  const [description, setDescription] = useState("");
  const [followups, setFollowups] = useState([]);
  const [pendingQuestions, setPendingQuestions] = useState([]);
  const [triageSummary, setTriageSummary] = useState("");
  const [triageField, setTriageField] = useState("");
  const [activePacks, setActivePacks] = useState([]);
  const [seedText, setSeedText] = useState("");
  const [followupRounds, setFollowupRounds] = useState(0);

  const [synthesis, setSynthesis] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState("");
  const [federationMeta, setFederationMeta] = useState(null);

  const runTriage = useCallback(async (desc, fups) => {
    setPhase("triaging");
    setError("");
    try {
      const res = await fetch("/api/triage", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ description: desc, followups: fups }),
      });
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.error || `triage failed (${res.status})`);
      }
      return await res.json();
    } catch (e) {
      setError(e.message || String(e));
      setPhase("error");
      return null;
    }
  }, []);

  const runSynthesis = useCallback(
    async (desc, fups, packs = []) => {
      setPhase("synthesizing");
      setError("");
      setSynthesis("");
      setFederationMeta(null);
      setStreaming(true);
      try {
        const res = await fetch("/api/synthesize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            description: desc,
            followups: fups,
            packs: packs.map((p) => p.id || p),
          }),
        });
        if (!res.ok) {
          const j = await res.json().catch(() => ({}));
          throw new Error(j.error || `synthesis failed (${res.status})`);
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let rawAcc = "";
        let renderedAcc = "";
        // Carry partial metadata sentinels across chunk boundaries.
        let metaBuffer = "";
        // eslint-disable-next-line no-constant-condition
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          rawAcc += chunk;
          metaBuffer += chunk;

          // Drain any complete metadata sentinels from metaBuffer.
          const { metas, stripped } = extractMetaAndStrip(metaBuffer);
          if (metas.length > 0) {
            // Apply each metadata update in order. The latest wins for
            // overlapping fields; failed_models stays present across phases.
            setFederationMeta((prev) => {
              let next = prev || {};
              for (const m of metas) next = { ...next, ...m };
              return next;
            });
          }

          // Compute the user-visible synthesis text: raw stream minus all
          // metadata sentinels seen so far. We do this on rawAcc (not on
          // stripped) so the displayed text remains coherent if a sentinel
          // arrives mid-chunk.
          renderedAcc = rawAcc.replace(META_RE, "");
          setSynthesis(renderedAcc);
          // metaBuffer keeps the trailing partial (everything after the
          // last fully-parsed sentinel); discard parsed prefix.
          metaBuffer = stripped;
        }
        setStreaming(false);
        setPhase("result");
        try {
          saveHistoryItem({
            description: desc,
            followups: fups,
            synthesis: renderedAcc,
            summary: triageSummary,
            field: triageField,
          });
        } catch {
          // ignore storage errors
        }
      } catch (e) {
        setStreaming(false);
        setError(e.message || String(e));
        setPhase("error");
      }
    },
    [triageSummary, triageField]
  );

  const handleInitialSubmit = useCallback(
    async (text) => {
      setDescription(text);
      setFollowups([]);
      setFollowupRounds(0);

      const result = await runTriage(text, []);
      if (!result) return;

      setTriageSummary(result.summary || "");
      setTriageField(result.field || "");
      const packs = result.packs || [];
      setActivePacks(packs);

      if (result.status === "ready") {
        runSynthesis(text, [], packs);
      } else {
        setPendingQuestions(result.questions || []);
        setPhase("followup");
      }
    },
    [runTriage, runSynthesis]
  );

  const handleFollowupSubmit = useCallback(
    async (newFollowups) => {
      const merged = [...followups, ...newFollowups];
      setFollowups(merged);
      const round = followupRounds + 1;
      setFollowupRounds(round);

      if (round >= MAX_FOLLOWUP_ROUNDS) {
        runSynthesis(description, merged, activePacks);
        return;
      }

      const result = await runTriage(description, merged);
      if (!result) return;
      setTriageSummary(result.summary || triageSummary);
      const packs = result.packs || activePacks;
      setActivePacks(packs);

      if (result.status === "ready") {
        runSynthesis(description, merged, packs);
      } else {
        setPendingQuestions(result.questions || []);
        setPhase("followup");
      }
    },
    [
      description,
      followups,
      followupRounds,
      runTriage,
      runSynthesis,
      triageSummary,
      activePacks,
    ]
  );

  const handleSkipFollowup = useCallback(() => {
    runSynthesis(description, followups, activePacks);
  }, [description, followups, activePacks, runSynthesis]);

  const handleReset = useCallback(() => {
    setPhase("input");
    setDescription("");
    setFollowups([]);
    setPendingQuestions([]);
    setSynthesis("");
    setError("");
    setTriageSummary("");
    setTriageField("");
    setActivePacks([]);
    setSeedText("");
    setFollowupRounds(0);
    setFederationMeta(null);
  }, []);

  return (
    <>
      <Head>
        <title>mechanistic-synthesis</title>
        <meta
          name="description"
          content="Describe an experiment; receive a procedural synthesis paper."
        />
      </Head>

      <div className="w-full min-h-[calc(100vh-180px)] px-8 sm:px-6 py-12">
        {phase === "input" && (
          <div className="max-w-3xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="mb-10"
            >
              <h1 className="text-3xl sm:text-2xl font-semibold tracking-tight text-dark dark:text-light mb-3">
                Describe an experiment.
              </h1>
              <p className="text-dark/60 dark:text-light/60 text-base leading-relaxed">
                Procedural-learning synthesis tool. Write what you&apos;re studying,
                what you&apos;re asking, what you plan to do — at whatever level of
                detail you have. The system reads it, asks clarifying questions if
                needed, and produces a paper-shaped synthesis: background, prior work,
                methods, expected results, statistics, pitfalls, and references.
              </p>
              <p className="text-xs text-dark/40 dark:text-light/40 mt-4 leading-relaxed">
                Each synthesis is drafted in parallel by a federation of three
                models, then merged by an integrator. Specialist knowledge packs
                activate automatically when the description matches their
                domain. Currently included:{" "}
                <span className="font-medium text-dark/60 dark:text-light/60">
                  cytochrome P450 — categorical mechanics
                </span>
                .
              </p>
            </motion.div>
            <ExperimentInput initial={seedText} onSubmit={handleInitialSubmit} />
            <ExampleQueries onPick={(t) => setSeedText(t)} />
          </div>
        )}

        {phase === "triaging" && <LoadingState phase="triaging" />}

        {phase === "followup" && (
          <div className="max-w-3xl mx-auto">
            <FollowupPanel
              summary={triageSummary}
              questions={pendingQuestions}
              packs={activePacks}
              onSubmit={handleFollowupSubmit}
              onSkip={handleSkipFollowup}
            />
          </div>
        )}

        {phase === "synthesizing" && synthesis.length === 0 && (
          <div className="max-w-3xl mx-auto">
            <FederationStatus meta={federationMeta} streaming />
            <LoadingState phase="synthesizing" />
          </div>
        )}

        {(phase === "synthesizing" || phase === "result") && synthesis.length > 0 && (
          <div className="w-full">
            <div className="max-w-6xl mx-auto mb-6 flex items-center justify-between gap-4">
              <button
                onClick={handleReset}
                className="text-sm text-dark/60 dark:text-light/60 hover:text-dark dark:hover:text-light transition shrink-0"
              >
                ← New synthesis
              </button>
              <div className="flex-1 flex justify-center">
                <PackBadge packs={activePacks} />
              </div>
              {phase === "result" ? (
                <div className="flex items-center gap-3 shrink-0">
                  <button
                    onClick={() => navigator.clipboard.writeText(synthesis)}
                    className="text-sm text-dark/60 dark:text-light/60 hover:text-dark dark:hover:text-light transition"
                  >
                    Copy markdown
                  </button>
                  <button
                    onClick={() => window.print()}
                    className="text-sm text-dark/60 dark:text-light/60 hover:text-dark dark:hover:text-light transition"
                  >
                    Print / PDF
                  </button>
                </div>
              ) : (
                <div className="shrink-0 w-[160px]" aria-hidden />
              )}
            </div>
            <div className="max-w-6xl mx-auto">
              <FederationStatus meta={federationMeta} streaming={streaming} />
            </div>
            <PaperRenderer markdown={synthesis} streaming={streaming} />
          </div>
        )}

        {phase === "error" && (
          <div className="max-w-3xl mx-auto py-12 text-center">
            <p className="text-primary dark:text-primaryDark font-medium mb-4">
              Something went wrong.
            </p>
            <p className="text-sm text-dark/70 dark:text-light/70 mb-6 font-mono">
              {error}
            </p>
            <button
              onClick={handleReset}
              className="px-5 py-2 rounded-md bg-dark text-light dark:bg-light dark:text-dark
                         font-medium hover:opacity-90 transition"
            >
              Start over
            </button>
          </div>
        )}
      </div>
    </>
  );
}
