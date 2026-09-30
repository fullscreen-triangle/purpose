import { useEffect, useState } from "react";
import { cancelJob, getJob, jobLog, listJobs } from "@/lib/purpose-client";
import { Button, Card, ErrorLine, Sparkline, StatusBadge, formatDuration } from "./ui";

const POLL_MS = 3000;

function usePoll(fn, deps, active = true) {
  useEffect(() => {
    if (!active) return undefined;
    let live = true;
    let timer;
    const tick = async () => {
      await fn(() => live);
      if (live) timer = setTimeout(tick, POLL_MS);
    };
    tick();
    return () => {
      live = false;
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}

function JobDetail({ conn, id }) {
  const [job, setJob] = useState(null);
  const [log, setLog] = useState([]);
  const [error, setError] = useState("");
  const finished = job && !["queued", "running"].includes(job.status);

  usePoll(
    async (live) => {
      try {
        const [j, l] = await Promise.all([getJob(conn, id), jobLog(conn, id, 60)]);
        if (live()) {
          setJob(j);
          setLog(l);
          setError("");
        }
      } catch (err) {
        if (live()) setError(err.message);
      }
    },
    [conn, id, finished],
    !finished,
  );

  if (!job) return <ErrorLine error={error} />;
  const p = job.progress;
  const now = Math.floor(Date.now() / 1000);
  const elapsed = job.started_at ? (job.finished_at || now) - job.started_at : null;
  const done = p ? p.epoch * p.steps_per_epoch + p.step : 0;
  const total = p ? p.epochs * p.steps_per_epoch : 0;
  const pct = total ? Math.min(100, (done / total) * 100) : 0;
  const eta = job.status === "running" && done && elapsed ? ((total - done) * elapsed) / done : null;

  return (
    <Card>
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <p className="text-base font-semibold text-dark dark:text-light">
            {job.theme} <StatusBadge status={job.status} />
          </p>
          <p className="text-xs text-dark/50 dark:text-light/50 mt-1 font-mono">{job.id}</p>
        </div>
        {!finished && (
          <Button variant="secondary" onClick={() => cancelJob(conn, id).then(setJob, (e) => setError(e.message))}>
            Cancel
          </Button>
        )}
      </div>

      <dl className="grid grid-cols-4 sm:grid-cols-2 gap-4 mt-5 text-sm">
        <div>
          <dt className="text-xs text-dark/50 dark:text-light/50">Stage</dt>
          <dd className="text-dark dark:text-light">{job.stage || "–"}</dd>
        </div>
        <div>
          <dt className="text-xs text-dark/50 dark:text-light/50">Progress</dt>
          <dd className="text-dark dark:text-light tabular-nums">
            {p ? `epoch ${p.epoch + 1}/${p.epochs} · step ${p.step}/${p.steps_per_epoch}` : "–"}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-dark/50 dark:text-light/50">Elapsed</dt>
          <dd className="text-dark dark:text-light tabular-nums">{formatDuration(elapsed)}</dd>
        </div>
        <div>
          <dt className="text-xs text-dark/50 dark:text-light/50">Remaining</dt>
          <dd className="text-dark dark:text-light tabular-nums">{eta == null ? "–" : `~${formatDuration(eta)}`}</dd>
        </div>
      </dl>

      {total > 0 && (
        <div className="mt-4 h-1.5 rounded-full bg-dark/10 dark:bg-light/10 overflow-hidden">
          <div className="h-full bg-primary dark:bg-primaryDark transition-all" style={{ width: `${pct}%` }} />
        </div>
      )}

      <div className="mt-5">
        <Sparkline values={job.loss_history} />
      </div>

      {job.error && <ErrorLine error={job.error} />}
      {job.result && (
        <p className="mt-4 text-sm text-dark/80 dark:text-light/80">
          {`Built from ${job.result.document_count} documents and ${job.result.example_count} examples. Find it under Models.`}
        </p>
      )}

      <pre className="mt-5 max-h-56 overflow-auto text-xs font-mono p-3 rounded-md bg-dark/5 dark:bg-light/5 text-dark/70 dark:text-light/70 whitespace-pre-wrap">
        {log.length ? log.join("\n") : "No log yet."}
      </pre>
      <ErrorLine error={error} />
    </Card>
  );
}

/** Every build job, newest first, with one selected for live detail. */
export default function JobsTab({ conn, selected, setSelected }) {
  const [jobs, setJobs] = useState([]);
  const [error, setError] = useState("");

  usePoll(
    async (live) => {
      try {
        const js = await listJobs(conn);
        if (live()) {
          setJobs(js);
          setError("");
        }
      } catch (err) {
        if (live()) setError(err.message);
      }
    },
    [conn],
  );

  const current = selected || jobs[0]?.id;

  return (
    <div className="grid grid-cols-[18rem_1fr] lg:grid-cols-1 gap-6">
      <Card title="Jobs">
        {jobs.length === 0 && <p className="text-sm text-dark/50 dark:text-light/50">No jobs yet.</p>}
        <ul className="space-y-1">
          {jobs.map((j) => (
            <li key={j.id}>
              <button
                onClick={() => setSelected(j.id)}
                className={`w-full text-left px-3 py-2 rounded-md text-sm transition ${
                  j.id === current ? "bg-dark/10 dark:bg-light/10" : "hover:bg-dark/5 dark:hover:bg-light/5"
                }`}
              >
                <span className="flex items-center justify-between gap-2">
                  <span className="truncate text-dark dark:text-light">{j.theme}</span>
                  <StatusBadge status={j.status} />
                </span>
                <span className="block text-xs text-dark/40 dark:text-light/40 mt-0.5">
                  {new Date(j.created_at * 1000).toLocaleString()}
                </span>
              </button>
            </li>
          ))}
        </ul>
        <ErrorLine error={error} />
      </Card>
      <div>{current && <JobDetail key={current} conn={conn} id={current} />}</div>
    </div>
  );
}
