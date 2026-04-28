import Head from "next/head";
import { useState, useCallback } from "react";
import { motion } from "framer-motion";

import ExperimentInput from "@/components/ExperimentInput";
import FollowupPanel from "@/components/FollowupPanel";
import PaperRenderer from "@/components/PaperRenderer";
import LoadingState from "@/components/LoadingState";
import { saveHistoryItem } from "@/lib/storage";

const MAX_FOLLOWUP_ROUNDS = 3;

export default function Home() {
  const [phase, setPhase] = useState("input");
  // "input" | "triaging" | "followup" | "synthesizing" | "result" | "error"

  const [description, setDescription] = useState("");
  const [followups, setFollowups] = useState([]);
  const [pendingQuestions, setPendingQuestions] = useState([]);
  const [triageSummary, setTriageSummary] = useState("");
  const [triageField, setTriageField] = useState("");
  const [followupRounds, setFollowupRounds] = useState(0);

  const [synthesis, setSynthesis] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState("");

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
    async (desc, fups) => {
      setPhase("synthesizing");
      setError("");
      setSynthesis("");
      setStreaming(true);
      try {
        const res = await fetch("/api/synthesize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ description: desc, followups: fups }),
        });
        if (!res.ok) {
          const j = await res.json().catch(() => ({}));
          throw new Error(j.error || `synthesis failed (${res.status})`);
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let acc = "";
        // eslint-disable-next-line no-constant-condition
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          acc += decoder.decode(value, { stream: true });
          setSynthesis(acc);
        }
        setStreaming(false);
        setPhase("result");
        try {
          saveHistoryItem({
            description: desc,
            followups: fups,
            synthesis: acc,
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

      if (result.status === "ready") {
        runSynthesis(text, []);
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
        runSynthesis(description, merged);
        return;
      }

      const result = await runTriage(description, merged);
      if (!result) return;
      setTriageSummary(result.summary || triageSummary);

      if (result.status === "ready") {
        runSynthesis(description, merged);
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
    ]
  );

  const handleSkipFollowup = useCallback(() => {
    runSynthesis(description, followups);
  }, [description, followups, runSynthesis]);

  const handleReset = useCallback(() => {
    setPhase("input");
    setDescription("");
    setFollowups([]);
    setPendingQuestions([]);
    setSynthesis("");
    setError("");
    setTriageSummary("");
    setTriageField("");
    setFollowupRounds(0);
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
            </motion.div>
            <ExperimentInput onSubmit={handleInitialSubmit} />
          </div>
        )}

        {phase === "triaging" && <LoadingState phase="triaging" />}

        {phase === "followup" && (
          <div className="max-w-3xl mx-auto">
            <FollowupPanel
              summary={triageSummary}
              questions={pendingQuestions}
              onSubmit={handleFollowupSubmit}
              onSkip={handleSkipFollowup}
            />
          </div>
        )}

        {phase === "synthesizing" && synthesis.length === 0 && (
          <LoadingState phase="synthesizing" />
        )}

        {(phase === "synthesizing" || phase === "result") && synthesis.length > 0 && (
          <div className="w-full">
            <div className="max-w-6xl mx-auto mb-6 flex items-center justify-between">
              <button
                onClick={handleReset}
                className="text-sm text-dark/60 dark:text-light/60 hover:text-dark dark:hover:text-light transition"
              >
                ← New synthesis
              </button>
              {phase === "result" && (
                <div className="flex items-center gap-3">
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
              )}
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
