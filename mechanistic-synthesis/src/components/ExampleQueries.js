import React from "react";
import { motion } from "framer-motion";

/**
 * Small panel of example experiment descriptions the user can click to
 * populate the textarea. Useful for first-time visitors who need to see
 * what kind of input the tool expects.
 */
const EXAMPLES = [
  {
    label: "CYP3A4 inhibition kinetics",
    text: `I'm planning a study of CYP3A4 inhibition by a new tetrahydroisoquinoline derivative. I have access to recombinant CYP3A4 expressed in baculovirus supersomes and would like to characterise the inhibition mode (competitive, non-competitive, mixed, mechanism-based). I'm worried about time-dependent inhibition because of a tertiary amine in the scaffold that might be N-dealkylated to a metabolic-intermediate complex. My plan is to run IC50 shifts with and without 30-minute NADPH preincubation, then dilution-recovery to test irreversibility. I'd like guidance on what to expect for this scaffold class and what additional experiments would distinguish MIC formation from heme alkylation.`,
  },
  {
    label: "Heme spin-state redox shift",
    text: `I want to measure how heme spin state controls redox potential in a CYP2D6 active site mutant. The mutant (Phe120Ala) is reported to shift the resting low-spin / high-spin equilibrium toward high-spin even without substrate. I plan to use spectroelectrochemistry under anaerobic conditions to pull a Nernst plot, and compare to wild-type. What size of E1/2 shift should I expect from a roughly 50% LS-to-HS shift, and what controls do I need to rule out confounding effects from the mutation itself rather than the spin redistribution?`,
  },
  {
    label: "P450 isoform deorphaning by sequence",
    text: `Our lab has a novel insect P450 that doesn't cluster cleanly with any characterised insect CYP family from blast results. We want to predict its likely substrate class and reaction repertoire from sequence alone before committing to expression. We have 484 amino acids, transmembrane prediction places it as membrane-anchored, and active-site residues align reasonably with CYP6 family. What sequence-level analyses would best constrain the prediction, and how confident can we be without crystallography?`,
  },
];

export default function ExampleQueries({ onPick }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.3 }}
      className="mt-8"
    >
      <p className="text-xs uppercase tracking-wider text-dark/45 dark:text-light/45 mb-3 font-medium">
        Or try one of these
      </p>
      <div className="grid grid-cols-1 gap-2.5">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.label}
            type="button"
            onClick={() => onPick(ex.text)}
            className="text-left px-4 py-3 rounded-md border border-dark/12 dark:border-light/12
                       hover:border-primary/40 dark:hover:border-primaryDark/40
                       transition group"
          >
            <p className="text-sm font-medium text-dark dark:text-light mb-1 group-hover:text-primary dark:group-hover:text-primaryDark transition">
              {ex.label}
            </p>
            <p className="text-xs text-dark/55 dark:text-light/50 line-clamp-2 leading-relaxed">
              {ex.text}
            </p>
          </button>
        ))}
      </div>
    </motion.div>
  );
}
