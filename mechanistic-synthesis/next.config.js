/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // Ensure the knowledge-packs/ directory is bundled into serverless
  // functions that read it at runtime (triage and synthesize routes).
  // Without this, Vercel's nft would strip the files because Next.js
  // cannot statically detect the runtime fs.readFileSync calls.
  experimental: {
    outputFileTracingIncludes: {
      "/api/synthesize": ["./knowledge-packs/**/*"],
      "/api/triage": ["./knowledge-packs/**/*"],
    },
  },
};

module.exports = nextConfig;
