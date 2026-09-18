import { join } from "node:path";
import { NextRequest, NextResponse } from "next/server";
import { Registry, http } from "@buhera/purpose-factory-client";
import { remoteServerConfig } from "@/lib/remote-config";

const REGISTRY_PATH = join(
  process.cwd(),
  "..",
  "implementation",
  ".purpose",
  "factory",
  "registry.json",
);

export async function GET(_req: NextRequest, { params }: { params: { id: string } }) {
  const remote = remoteServerConfig();
  const models = remote
    ? await http.listThemes(remote)
    : await Registry.atPath(REGISTRY_PATH).load();
  return NextResponse.json(models.filter((m) => m.name === params.id));
}
