import { ArchitectureFlowDiagram } from "@/components/explainer/ArchitectureFlowDiagram";
import { ArchitectureGraph } from "@/components/explainer/ArchitectureGraph";
import { ConnectMainModelExample } from "@/components/explainer/ConnectMainModelExample";
import { Hero } from "@/components/explainer/Hero";
import { HonestyPanel } from "@/components/explainer/HonestyPanel";
import { LayerCards } from "@/components/explainer/LayerCards";
import { PartitionExtinctionScene } from "@/components/explainer/PartitionExtinctionScene";
import { PhaseLockOrderSurface } from "@/components/explainer/PhaseLockOrderSurface";
import { QueryFlowStepper } from "@/components/explainer/QueryFlowStepper";

export default function LandingPage() {
  return (
    <div className="-mx-6 -my-10 bg-dark px-6 py-10 text-light">
      <div className="mx-auto max-w-5xl">
        <Hero />

        <section id="how-it-works" className="mt-8">
          <h2 className="text-xl font-semibold">The problem with content-store RAG</h2>
          <p className="mt-2 max-w-2xl text-sm text-light/60">
            Most systems make one big model swallow everyone&rsquo;s documents, or retrieve raw
            text from them at query time. Drag the nodes below — only the diagram on the right
            keeps content where it belongs.
          </p>
          <div className="mt-6">
            <ArchitectureGraph />
          </div>
        </section>

        <section className="mt-16">
          <h2 className="text-xl font-semibold">How a query flows</h2>
          <p className="mt-2 max-w-2xl text-sm text-light/60">
            Step through what happens between you asking a question and getting an answer.
          </p>
          <div className="mt-6 max-w-md">
            <QueryFlowStepper />
          </div>
        </section>

        <section className="mt-16">
          <h2 className="text-xl font-semibold">The three layers</h2>
          <p className="mt-2 max-w-2xl text-sm text-light/60">
            Each layer is a proven theorem, not a heuristic. Click a card for the detail.
          </p>
          <div className="mt-6">
            <LayerCards />
          </div>
        </section>

        <section className="mt-16">
          <h2 className="text-xl font-semibold">Why this is honest, not magic</h2>
          <div className="mt-6 max-w-lg">
            <HonestyPanel />
          </div>
        </section>

        <section className="mt-16">
          <h2 className="text-xl font-semibold">Why locking is binary, not a threshold</h2>
          <p className="mt-2 max-w-2xl text-sm text-light/60">
            Earlier we said profiles either lock or they don&rsquo;t. Here&rsquo;s
            why that&rsquo;s not a design choice — it&rsquo;s a discontinuity in the
            underlying quantity, so there was never an intermediate state to
            average against.
          </p>
          <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
            <PartitionExtinctionScene />
            <PhaseLockOrderSurface />
          </div>
        </section>

        <section className="mt-16">
          <h2 className="text-xl font-semibold">The full architecture</h2>
          <p className="mt-2 max-w-2xl text-sm text-light/60">
            Every stage below is an ordinary receiver operation, governed by
            the same composable floor. Hover or click a stage for the theorem
            behind it.
          </p>
          <div className="mt-6">
            <ArchitectureFlowDiagram />
          </div>
        </section>

        <section className="mt-16">
          <h2 className="text-xl font-semibold">Connecting to the main model</h2>
          <p className="mt-2 max-w-2xl text-sm text-light/60">
            A minimal, worked example: one query, three illustrative
            profiles, and what actually gets combined.
          </p>
          <div className="mt-6 max-w-2xl">
            <ConnectMainModelExample />
          </div>
        </section>

        <footer className="mt-20 border-t border-white/10 pt-8 pb-4 text-sm text-light/50">
          <p>
            Read the paper:{" "}
            <span className="text-light/70">
              absicht/docs/distributed-domain-route-graph/distributed-domain-route-graph.tex
            </span>
          </p>
          <p className="mt-2">
            Ready to try it?{" "}
            <a href="/profiles" className="text-primaryDark hover:underline">
              Create a profile →
            </a>
          </p>
        </footer>
      </div>
    </div>
  );
}
