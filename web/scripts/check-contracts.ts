import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { CONFIG_NAMES, ContractError, loadBytes, validateArtifact, validateConfigBundle,
  type ConfigBundle } from "../src/contracts/loader";
import type { SchemaName } from "../src/contracts/schemas";

const args = process.argv.slice(2);
function option(name: string, fallback?: string) {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : fallback;
}
const shared = option("--shared", fileURLToPath(new URL("../../shared", import.meta.url)))!;
const result: Record<string, unknown> = {};
try {
  for (const name of CONFIG_NAMES) result[name] = await loadBytes(name, await readFile(resolve(shared, `${name}.v1.json`)));
  validateConfigBundle(result as ConfigBundle);
  const fixturePath = option("--fixtures");
  if (fixturePath) {
    const cases = JSON.parse(await readFile(fixturePath, "utf8")) as { id: string; schema: SchemaName; data: unknown }[];
    result.fixtures = cases.map(test => {
      try { validateArtifact(test.schema, test.data); return { id: test.id, accepted: true }; }
      catch (error) { if (!(error instanceof ContractError)) throw error; return { id: test.id, accepted: false }; }
    });
  }
  console.log(JSON.stringify(result));
} catch (error) { console.error(String(error)); process.exitCode = 2; }
