import Head from "next/head";
import { useEffect, useState } from "react";
import ConnectPanel from "@/components/platform/ConnectPanel";
import JobsTab from "@/components/platform/JobsTab";
import ModelsTab from "@/components/platform/ModelsTab";
import PlanTab from "@/components/platform/PlanTab";
import TrainTab from "@/components/platform/TrainTab";
import { DEFAULT_URL, health, loadConnection } from "@/lib/purpose-client";

const TABS = ["Plan", "Train", "Jobs", "Models"];
const HEALTH_MS = 10_000;

function repoFor(paramsB) {
  const size = { 0.5: "0.5B", 1.5: "1.5B", 3: "3B", 7: "7B", 14: "14B", 32: "32B" }[paramsB];
  return size ? `Qwen/Qwen2.5-${size}-Instruct` : "Qwen/Qwen2.5-0.5B-Instruct";
}

export default function Platform() {
  const [conn, setConn] = useState({ url: DEFAULT_URL, token: "" });
  const [loaded, setLoaded] = useState(false);
  const [status, setStatus] = useState({ ok: false, checking: true });
  const [tab, setTab] = useState("Plan");
  const [preset, setPreset] = useState({});
  const [selectedJob, setSelectedJob] = useState(null);

  useEffect(() => {
    setConn(loadConnection());
    setLoaded(true);
  }, []);

  useEffect(() => {
    if (!loaded) return undefined;
    if (!conn.token) {
      setStatus({ ok: false, checking: false });
      return undefined;
    }
    let live = true;
    const check = () =>
      health(conn).then(
        (h) => live && setStatus({ ok: true, version: h.version, jobRunning: h.job_running }),
        (e) => live && setStatus({ ok: false, error: e.message }),
      );
    setStatus({ ok: false, checking: true });
    check();
    const t = setInterval(check, HEALTH_MS);
    return () => {
      live = false;
      clearInterval(t);
    };
  }, [conn, loaded]);

  function applyPlan(form) {
    setPreset({
      kind: "pretrained",
      repo: repoFor(Number(form.params_b)),
      epochs: Number(form.epochs),
      block_size: Math.min(Number(form.seq_len), 256),
      batch_size: 1,
    });
    setTab("Train");
  }

  return (
    <>
      <Head>
        <title>Model platform · mechanistic-synthesis</title>
      </Head>
      <div className="w-full min-h-[calc(100vh-180px)] px-8 sm:px-6 py-12">
        <div className="max-w-6xl mx-auto space-y-6">
          <div>
            <h1 className="text-3xl sm:text-2xl font-semibold tracking-tight text-dark dark:text-light mb-2">
              Model platform
            </h1>
            <p className="text-dark/60 dark:text-light/60 text-base leading-relaxed max-w-3xl">
              Plan where and how to train, upload your material, and run and watch training on this
              laptop through your local <code className="font-mono text-sm">purpose</code>. Nothing
              here leaves this machine except through a place you choose.
            </p>
          </div>

          {loaded && <ConnectPanel conn={conn} setConn={setConn} status={status} />}

          {status.ok && (
            <>
              <nav className="flex gap-1 border-b border-dark/10 dark:border-light/10" aria-label="Platform sections">
                {TABS.map((t) => (
                  <button
                    key={t}
                    onClick={() => setTab(t)}
                    aria-current={tab === t ? "page" : undefined}
                    className={`px-4 py-2 text-sm font-medium -mb-px border-b-2 transition ${
                      tab === t
                        ? "border-primary dark:border-primaryDark text-dark dark:text-light"
                        : "border-transparent text-dark/50 dark:text-light/50 hover:text-dark dark:hover:text-light"
                    }`}
                  >
                    {t}
                  </button>
                ))}
              </nav>

              {tab === "Plan" && <PlanTab conn={conn} onUse={applyPlan} />}
              {tab === "Train" && (
                <TrainTab
                  conn={conn}
                  preset={preset}
                  onStarted={(job) => {
                    setSelectedJob(job.id);
                    setTab("Jobs");
                  }}
                />
              )}
              {tab === "Jobs" && <JobsTab conn={conn} selected={selectedJob} setSelected={setSelectedJob} />}
              {tab === "Models" && <ModelsTab conn={conn} />}
            </>
          )}
        </div>
      </div>
    </>
  );
}
