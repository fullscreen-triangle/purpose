// Small shared pieces for the platform tabs, in the site's existing idiom.

export function Card({ title, children, className = "" }) {
  return (
    <section className={`rounded-md border border-dark/10 dark:border-light/10 bg-dark/[0.02] dark:bg-light/[0.03] p-5 ${className}`}>
      {title && <h2 className="text-sm font-semibold text-dark dark:text-light mb-4">{title}</h2>}
      {children}
    </section>
  );
}

export function Field({ label, hint, children }) {
  return (
    <label className="block">
      <span className="block text-xs font-medium text-dark/70 dark:text-light/70 mb-1">{label}</span>
      {children}
      {hint && <span className="block text-xs text-dark/40 dark:text-light/40 mt-1">{hint}</span>}
    </label>
  );
}

export const inputClass =
  "w-full px-3 py-2 rounded-md bg-dark/5 dark:bg-light/5 border border-dark/10 dark:border-light/10 " +
  "text-sm text-dark dark:text-light focus:outline-none focus:border-primary dark:focus:border-primaryDark";

export function Button({ children, variant = "primary", className = "", ...props }) {
  const look =
    variant === "primary"
      ? "bg-dark text-light dark:bg-light dark:text-dark"
      : "bg-dark/5 dark:bg-light/10 text-dark dark:text-light hover:bg-dark/10 dark:hover:bg-light/20";
  return (
    <button
      {...props}
      className={`px-4 py-2 rounded-md text-sm font-medium transition hover:opacity-90 disabled:opacity-40 ${look} ${className}`}
    >
      {children}
    </button>
  );
}

export function ErrorLine({ error }) {
  if (!error) return null;
  return (
    <p role="alert" className="mt-3 text-sm text-primary dark:text-primaryDark">
      {error}
    </p>
  );
}

const STATUS_STYLE = {
  queued: "bg-dark/10 dark:bg-light/10 text-dark/70 dark:text-light/70",
  running: "bg-primaryDark/20 text-dark dark:text-primaryDark",
  succeeded: "bg-primaryDark/30 text-dark dark:text-light",
  failed: "bg-primary/20 text-primary dark:text-primary",
  cancelled: "bg-dark/10 dark:bg-light/10 text-dark/60 dark:text-light/60",
  interrupted: "bg-primary/10 text-primary",
};

export function StatusBadge({ status }) {
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${STATUS_STYLE[status] || ""}`}>
      {status}
    </span>
  );
}

/** Loss curve as an inline SVG line; y is scaled to the data's own range. */
export function Sparkline({ values, width = 320, height = 64 }) {
  if (!values || values.length < 2) {
    return <p className="text-xs text-dark/40 dark:text-light/40">No loss points yet.</p>;
  }
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const pts = values
    .map((v, i) => `${(i / (values.length - 1)) * width},${height - 4 - ((v - lo) / span) * (height - 8)}`)
    .join(" ");
  return (
    <figure>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-16" role="img" aria-label="training loss">
        <polyline points={pts} fill="none" strokeWidth="2" className="stroke-primary dark:stroke-primaryDark" />
      </svg>
      <figcaption className="text-xs text-dark/50 dark:text-light/50 mt-1">
        loss {values[0].toFixed(3)} → {values[values.length - 1].toFixed(3)}
      </figcaption>
    </figure>
  );
}

export function formatDuration(seconds) {
  if (seconds == null) return "–";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return h ? `${h}h ${m}m` : m ? `${m}m ${s}s` : `${s}s`;
}
