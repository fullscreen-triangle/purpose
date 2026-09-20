import Link from "next/link";

export function Hero() {
  return (
    <div className="py-16 text-center">
      <h1 className="text-4xl font-bold tracking-tight text-light">
        A Federation of Profiles
      </h1>
      <p className="mx-auto mt-4 max-w-xl text-light/70">
        Your profile trains its own small model on what you feed it. When you ask something, the
        main model doesn&rsquo;t try to know everything itself — it asks the right profiles, the
        way you&rsquo;d ask the right expert, and combines honest answers into one.
      </p>
      <div className="mt-8 flex justify-center gap-3">
        <Link
          href="/profiles"
          className="rounded bg-primary px-5 py-2.5 text-sm font-medium text-white transition hover:opacity-90"
        >
          Use Purpose →
        </Link>
        <a
          href="#how-it-works"
          className="rounded border border-white/20 px-5 py-2.5 text-sm font-medium text-light/80 transition hover:border-primaryDark/50"
        >
          See how it works
        </a>
      </div>
    </div>
  );
}
