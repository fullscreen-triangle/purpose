import { useEffect, useState } from "react";
import { downloadModelFile, listModels } from "@/lib/purpose-client";
import { Button, Card, ErrorLine } from "./ui";

const FILES = ["model.safetensors", "config.json", "tokenizer.json"];

/** Built models from purpose's registry, with their files to download. */
export default function ModelsTab({ conn }) {
  const [models, setModels] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    listModels(conn).then(setModels, (e) => setError(e.message));
  }, [conn]);

  return (
    <Card title="Models">
      {models && models.length === 0 && (
        <p className="text-sm text-dark/50 dark:text-light/50">No models built yet. Start one under Train.</p>
      )}
      <ul className="divide-y divide-dark/10 dark:divide-light/10">
        {(models || []).map((m) => (
          <li key={m.name} className="py-4 flex items-start justify-between gap-4 flex-wrap">
            <div className="min-w-0">
              <p className="text-dark dark:text-light font-medium">{m.name}</p>
              <p className="text-xs text-dark/50 dark:text-light/50 mt-1">
                {`${m.document_count} documents · ${m.example_count} examples · vocabulary ${m.vocab_size.toLocaleString()}`}
              </p>
              <p className="text-xs text-dark/40 dark:text-light/40 font-mono mt-1 truncate">{m.path}</p>
            </div>
            <div className="flex gap-2 flex-wrap">
              {FILES.map((f) => (
                <Button
                  key={f}
                  variant="secondary"
                  onClick={() => downloadModelFile(conn, m.name, f).catch((e) => setError(e.message))}
                >
                  {f}
                </Button>
              ))}
            </div>
          </li>
        ))}
      </ul>
      <ErrorLine error={error} />
    </Card>
  );
}
