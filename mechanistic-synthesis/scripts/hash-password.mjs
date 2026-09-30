// Prints the two env vars the site's login needs. Run locally:
//
//     node scripts/hash-password.mjs
//
// It asks for the password without echoing it, and prints
//   PLATFORM_PASSWORD_HASH=scrypt$...   (safe to store: it is a hash)
//   PLATFORM_SESSION_SECRET=...         (random; keep it secret)
// Put both in .env.local and in the Vercel project's environment variables.

import { randomBytes } from "node:crypto";
import readline from "node:readline";
import { scryptAsync } from "../src/lib/password.js";

function ask(question) {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout, terminal: true });
  let muted = false;
  rl._writeToOutput = (s) => {
    if (!muted) rl.output.write(s);
  };
  return new Promise((resolve) => {
    rl.question(question, (answer) => {
      rl.close();
      process.stdout.write("\n");
      resolve(answer);
    });
    muted = true;
  });
}

const password = await ask("Password: ");
const again = await ask("Again:    ");
if (password !== again) {
  console.error("The two entries differ; nothing printed.");
  process.exit(1);
}
if (password.length < 12) {
  console.error("Use at least 12 characters; nothing printed.");
  process.exit(1);
}

const [N, r, p] = [16384, 8, 1];
const salt = randomBytes(16);
const hash = await scryptAsync(password, salt, 64, { N, r, p });
console.log(`\nPLATFORM_PASSWORD_HASH=scrypt$${N}$${r}$${p}$${salt.toString("base64")}$${hash.toString("base64")}`);
console.log(`PLATFORM_SESSION_SECRET=${randomBytes(32).toString("hex")}`);
