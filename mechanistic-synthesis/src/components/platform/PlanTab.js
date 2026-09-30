import { useState } from "react";
import { planRun } from "@/lib/purpose-client";
import { Button, Card, ErrorLine, Field, inputClass } from "./ui";

const SIZES = [0.5, 1.5, 3, 7, 14, 32];

/** "400k", "1.2M", "2e6" → number. */
export function parseCount(s) {
  const t = String(s).trim().toLowerCase();
  const mul = { k: 1e3, m: 1e6, b: 1e9 }[t.slice(-1)] || 1;
  const n = parseFloat(mul === 1 ? t : t.slice(0, -1));
  return Number.isFinite(n) ? n * mul : NaN;
}

const eur = (v) => (v === 0 ? "free" : `€${v.toFixed(2)}`);
const hours = (h) => (h < 1 ? `${Math.max(1, Math.round(h * 60))} min` : `${h.toFixed(1)} h`);

/** Ranks every place to train for one run, and says how. */
export default function PlanTab({ conn, onUse }) {
  const [form, setForm] = useState({
    params_b: 0.5,
    tokens: "400k",
    epochs: 3,
    seq_len: 1024,
    privacy: "internal",
    goal: "vocabulary",
    method: "auto",
  });
  const [plan, setPlan] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));

  async function submit(e) {
    e.preventDefault();
    const tokens = parseCount(form.tokens);
    if (!(tokens > 0)) return setError("Tokens must be a number, e.g. 400k or 1.2M.");
    setBusy(true);
    setError("");
    try {
      setPlan(
        await planRun(conn, {
          ...form,
          params_b: Number(form.params_b),
          tokens,
          epochs: Number(form.epochs),
          seq_len: Number(form.seq_len),
        }),
      );
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <Card title="The run">
        <form onSubmit={submit} className="grid grid-cols-4 lg:grid-cols-2 sm:grid-cols-1 gap-4">
          <Field label="Model size">
            <select className={inputClass} value={form.params_b} onChange={set("params_b")}>
              {SIZES.map((s) => (
                <option key={s} value={s}>{`${s}B parameters`}</option>
              ))}
            </select>
          </Field>
          <Field label="Training tokens" hint="Per epoch. About 4 characters per token.">
            <input className={inputClass} value={form.tokens} onChange={set("tokens")} />
          </Field>
          <Field label="Epochs">
            <input className={inputClass} type="number" min="1" max="50" value={form.epochs} onChange={set("epochs")} />
          </Field>
          <Field label="Sequence length">
            <select className={inputClass} value={form.seq_len} onChange={set("seq_len")}>
              {[256, 512, 1024, 2048, 4096].map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </Field>
          <Field label="The data is">
            <select className={inputClass} value={form.privacy} onChange={set("privacy")}>
              <option value="public">public</option>
              <option value="internal">internal (work, not personal)</option>
              <option value="private">private (personal data)</option>
            </select>
          </Field>
          <Field label="The model should learn">
            <select className={inputClass} value={form.goal} onChange={set("goal")}>
              <option value="vocabulary">the field&apos;s language and reasoning</option>
              <option value="tasks">a task, from worked examples</option>
              <option value="facts">specific facts and numbers</option>
            </select>
          </Field>
          <Field label="Method">
            <select className={inputClass} value={form.method} onChange={set("method")}>
              <option value="auto">choose for me</option>
              <option value="lora">LoRA</option>
              <option value="qlora">QLoRA</option>
              <option value="full">full fine-tune</option>
            </select>
          </Field>
          <div className="flex items-end">
            <Button type="submit" disabled={busy} className="w-full">
              {busy ? "Planning…" : "Plan"}
            </Button>
          </div>
        </form>
        <ErrorLine error={error} />
      </Card>

      {plan && (
        <>
          <Card title={`Approach: ${plan.advice.approach}`}>
            <ul className="space-y-2 text-sm text-dark/80 dark:text-light/80 list-disc pl-5">
              {plan.advice.reasons.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          </Card>

          <Card title="Where to train">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs text-dark/50 dark:text-light/50 border-b border-dark/10 dark:border-light/10">
                    <th className="py-2 pr-4 font-medium">Place</th>
                    <th className="py-2 pr-4 font-medium">Method</th>
                    <th className="py-2 pr-4 font-medium text-right">Memory</th>
                    <th className="py-2 pr-4 font-medium text-right">Time</th>
                    <th className="py-2 pr-4 font-medium text-right">Cost</th>
                    <th className="py-2 font-medium" />
                  </tr>
                </thead>
                <tbody>
                  {plan.options.map((o) => (
                    <tr key={o.place} className="border-b border-dark/5 dark:border-light/5 align-top">
                      <td className="py-3 pr-4">
                        <p className={o.feasible ? "text-dark dark:text-light" : "text-dark/40 dark:text-light/40"}>
                          {o.name}
                        </p>
                        {o.feasible ? (
                          <p className="text-xs text-dark/50 dark:text-light/50 mt-1">{o.how_to}</p>
                        ) : (
                          <p className="text-xs text-primary dark:text-primaryDark mt-1">{o.why_not.join("; ")}</p>
                        )}
                      </td>
                      <td className="py-3 pr-4 uppercase text-xs">{o.method}</td>
                      <td className="py-3 pr-4 text-right tabular-nums">{o.memory_gb} GB</td>
                      <td className="py-3 pr-4 text-right tabular-nums">{hours(o.hours)}</td>
                      <td className="py-3 pr-4 text-right tabular-nums">{eur(o.cost_eur)}</td>
                      <td className="py-3 text-right">
                        {o.feasible && o.automatic && (
                          <Button variant="secondary" onClick={() => onUse(form, o)}>
                            Train here
                          </Button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-dark/40 dark:text-light/40 mt-4">
              Estimates: memory from model shape and sequence length; time from 6·N·T FLOPs at a
              realistic fraction of each GPU&apos;s peak (the laptop is calibrated to a measured run).
              Edit prices in <code className="font-mono">places.toml</code> in purpose&apos;s root.
            </p>
          </Card>
        </>
      )}
    </div>
  );
}
