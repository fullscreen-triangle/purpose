/** @type {import('next').NextConfig} */
const nextConfig = {
  // purpose-factory-ts spawns the `purpose` binary and touches the local
  // filesystem — server-only, never bundled for the client. Next.js 14 key
  // (experimental.serverComponentsExternalPackages); renamed to the
  // top-level serverExternalPackages in Next.js 15 — update this if/when
  // this app upgrades, since the old key is silently ignored on 15+.
  experimental: {
    serverComponentsExternalPackages: ["@buhera/purpose-factory-client"],
  },
  eslint: {
    // This app lives nested under mechanistic-synthesis/, which has its own
    // .eslintrc.json; ESLint's legacy config resolution picks up both and
    // reports a plugin conflict during `next build`'s lint step even though
    // the actual build succeeds. Run `npx next lint` manually when wanted;
    // don't let a monorepo-nesting quirk fail production builds.
    ignoreDuringBuilds: true,
  },
};

export default nextConfig;
