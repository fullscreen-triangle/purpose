import React from "react";
import { motion } from "framer-motion";

/**
 * Visualises the FKAC pipeline state during synthesis: which draft models
 * are running, which have completed, and the aggregate floor/confidence
 * estimate once integration is under way.
 */
function shortName(id) {
  if (!id) return "";
  // strip org prefix: "meta-llama/Llama-3.1-8B-Instruct" -> "Llama-3.1-8B"
  const parts = id.split("/");
  const last = parts[parts.length - 1];
  return last.replace(/-Instruct.*$/i, "").replace(/_/g, " ");
}

export default function FederationStatus({ meta, streaming }) {
  if (!meta) return null;
  const phase = meta.phase;
  const draftModels = meta.draft_models || [];
  const failed = new Set(meta.failed_models || []);

  if (phase === "single") {
    return (
      <div className="text-xs text-dark/55 dark:text-light/55 mb-2">
        single model: <span className="font-medium">{shortName(draftModels[0])}</span>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: -6 }}
      animate={{ opacity: 1, y: 0 }}
      className="mb-3 px-4 py-3 rounded-md bg-dark/5 dark:bg-light/5
                 border border-dark/10 dark:border-light/10"
    >
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div className="flex flex-col gap-1">
          <p className="text-xs uppercase tracking-wider text-dark/55 dark:text-light/55">
            {phase === "drafting"
              ? "Federation drafting"
              : phase === "integrating"
              ? "Integrating drafts"
              : "Federation"}
          </p>
          <div className="flex flex-wrap gap-2">
            {draftModels.map((id) => {
              const isFailed = failed.has(id);
              return (
                <span
                  key={id}
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs
                    ${
                      isFailed
                        ? "bg-primary/10 dark:bg-primaryDark/10 text-primary/80 dark:text-primaryDark/80 line-through"
                        : "bg-primary/15 dark:bg-primaryDark/20 text-primary dark:text-primaryDark"
                    }`}
                >
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      isFailed
                        ? "bg-primary/40"
                        : phase === "drafting"
                        ? "bg-primary animate-pulse"
                        : "bg-primary"
                    }`}
                    aria-hidden
                  />
                  {shortName(id)}
                </span>
              );
            })}
          </div>
        </div>

        {phase === "integrating" && meta.confidence != null && (
          <div className="flex flex-col items-end">
            <p className="text-[10px] uppercase tracking-wider text-dark/45 dark:text-light/45">
              aggregate floor
            </p>
            <p className="text-sm font-medium text-dark dark:text-light tabular-nums">
              {meta.aggregate_floor.toFixed(2)} / 100
            </p>
            <p className="text-[10px] text-dark/55 dark:text-light/55 tabular-nums">
              confidence {(meta.confidence * 100).toFixed(1)}%
            </p>
          </div>
        )}
      </div>
    </motion.div>
  );
}
