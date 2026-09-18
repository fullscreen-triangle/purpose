// Configuration for talking to a remote `purpose serve` server instead of
// spawning `purpose` as a local subprocess. Set both env vars to switch a
// profile-web deployment to the network transport; leave them unset to keep
// the default local-subprocess behavior (profile-web and purpose-factory on
// the same machine, sharing a filesystem).

export interface RemoteServerConfig {
  baseUrl: string;
  token: string;
}

export function remoteServerConfig(): RemoteServerConfig | null {
  const baseUrl = process.env.PURPOSE_SERVE_URL;
  const token = process.env.PURPOSE_SERVE_TOKEN;
  if (!baseUrl || !token) return null;
  return { baseUrl, token };
}
