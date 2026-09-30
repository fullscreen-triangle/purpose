import { useEffect, useState } from "react";
import { listSources, submitJob, uploadSources } from "@/lib/purpose-client";
import { Button, Card, ErrorLine, Field, inputClass } from "./ui";

const ACCEPT = ".tex,.pdf,.md,.txt,.csv,.json";
const THEME_RE = /^[A-Za-z0-9_-][A-Za-z0-9._-]{0,63}$/;

export const DEFAULT_TRAINING = {
  theme: "",
  kind: "pretrained",
  repo: "Qwen/Qwen2.5-0.5B-Instruct",
  block_size: 256,
  epochs: 3,
  batch_size: 1,
  learning_rate: 2e-4,
  lora_rank: 16,
  lora_alpha: 32,
};

/** Upload material for a theme and start a build job on the laptop. */
export default function TrainTab({ conn, preset, onStarted }) {
  const [cfg, setCfg] = useState({ ...DEFAULT_TRAINING, ...preset });
  const [files, setFiles] = useState([]);
  const [sources, setSources] = useState([]);
  const [upload, setUpload] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");
  const set = (k) => (e) => setCfg((c) => ({ ...c, [k]: e.target.value }));
  const themeOk = THEME_RE.test(cfg.theme);

  useEffect(() => setCfg((c) => ({ ...c, ...preset })), [preset]);

  useEffect(() => {
    if (!themeOk) return setSources([]);
    let live = true;
    listSources(conn, cfg.theme)
      .then((s) => live && setSources(s))
      .catch(() => live && setSources([]));
    return () => {
      live = false;
    };
  }, [conn, cfg.theme, themeOk, upload]);

  async function doUpload() {
    setBusy("upload");
    setError("");
    try {
      setUpload(await uploadSources(conn, cfg.theme, files));
      setFiles([]);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  async function start(e) {
    e.preventDefault();
    setBusy("start");
    setError("");
    try {
      const model =
        cfg.kind === "scratch"
          ? { kind: "scratch" }
          : { kind: "pretrained", repo: cfg.repo.trim(), block_size: Number(cfg.block_size) };
      const job = await submitJob(conn, {
        theme: cfg.theme,
        model,
        training: {
          epochs: Number(cfg.epochs),
          batch_size: Number(cfg.batch_size),
          learning_rate: Number(cfg.learning_rate),
          lora_rank: Number(cfg.lora_rank),
          lora_alpha: Number(cfg.lora_alpha),
        },
      });
      onStarted(job);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  const totalBytes = sources.reduce((n, s) => n + s.bytes, 0);

  return (
    <form onSubmit={start} className="space-y-6">
      <Card title="1 · Theme and material">
        <div className="grid grid-cols-2 sm:grid-cols-1 gap-4">
          <Field label="Theme name" hint="Letters, digits, - _ . — becomes the model's name.">
            <input className={inputClass} value={cfg.theme} onChange={set("theme")} placeholder="e.g. absicht" />
          </Field>
          <Field label="Add files" hint={`${ACCEPT.replaceAll(",", " ")} · up to 25 MB each`}>
            <input
              type="file"
              multiple
              accept={ACCEPT}
              onChange={(e) => setFiles(Array.from(e.target.files || []))}
              className="block w-full text-sm text-dark/70 dark:text-light/70"
            />
          </Field>
        </div>
        <div className="mt-4 flex items-center gap-4 flex-wrap">
          <Button
            type="button"
            variant="secondary"
            disabled={!themeOk || !files.length || busy === "upload"}
            onClick={doUpload}
          >
            {busy === "upload" ? "Uploading…" : `Upload ${files.length || ""} file${files.length === 1 ? "" : "s"}`}
          </Button>
          {themeOk && (
            <span className="text-xs text-dark/50 dark:text-light/50">
              {sources.length
                ? `${sources.length} file${sources.length === 1 ? "" : "s"} in this theme, ${(totalBytes / 1024).toFixed(0)} KB`
                : "No material in this theme yet."}
            </span>
          )}
        </div>
        {upload?.rejected?.length > 0 && (
          <ul className="mt-3 text-xs text-primary dark:text-primaryDark">
            {upload.rejected.map((r) => (
              <li key={r.filename}>{`${r.filename}: ${r.reason}`}</li>
            ))}
          </ul>
        )}
        {sources.length > 0 && (
          <ul className="mt-3 text-xs text-dark/60 dark:text-light/60 font-mono max-h-32 overflow-y-auto">
            {sources.map((s) => (
              <li key={s.filename}>{s.filename}</li>
            ))}
          </ul>
        )}
      </Card>

      <Card title="2 · Model and training">
        <div className="grid grid-cols-3 lg:grid-cols-2 sm:grid-cols-1 gap-4">
          <Field label="Start from">
            <select className={inputClass} value={cfg.kind} onChange={set("kind")}>
              <option value="pretrained">a pretrained model (LoRA)</option>
              <option value="scratch">scratch (tiny test model)</option>
            </select>
          </Field>
          {cfg.kind === "pretrained" && (
            <>
              <Field label="Hugging Face model" hint="Qwen2 or LLaMA family, single safetensors file.">
                <input className={inputClass} value={cfg.repo} onChange={set("repo")} />
              </Field>
              <Field label="Block size (tokens)" hint="256 fits 0.5B in this laptop's memory.">
                <input className={inputClass} type="number" min="32" max="4096" value={cfg.block_size} onChange={set("block_size")} />
              </Field>
            </>
          )}
          <Field label="Epochs">
            <input className={inputClass} type="number" min="1" max="50" value={cfg.epochs} onChange={set("epochs")} />
          </Field>
          <Field label="Batch size">
            <input className={inputClass} type="number" min="1" max="64" value={cfg.batch_size} onChange={set("batch_size")} />
          </Field>
          <Field label="Learning rate">
            <input className={inputClass} value={cfg.learning_rate} onChange={set("learning_rate")} />
          </Field>
          <Field label="LoRA rank">
            <input className={inputClass} type="number" min="1" max="256" value={cfg.lora_rank} onChange={set("lora_rank")} />
          </Field>
          <Field label="LoRA alpha">
            <input className={inputClass} type="number" min="1" max="512" value={cfg.lora_alpha} onChange={set("lora_alpha")} />
          </Field>
        </div>
        <div className="mt-6 flex items-center gap-4">
          <Button type="submit" disabled={!themeOk || !sources.length || busy === "start"}>
            {busy === "start" ? "Starting…" : "Start training on this laptop"}
          </Button>
          {!sources.length && themeOk && (
            <span className="text-xs text-dark/50 dark:text-light/50">Upload material first.</span>
          )}
        </div>
        <ErrorLine error={error} />
      </Card>
    </form>
  );
}
