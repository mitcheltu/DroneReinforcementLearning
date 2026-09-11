import { existsSync, renameSync, rmSync } from "node:fs";
import { spawnSync } from "node:child_process";
import path from "node:path";

const apiDirectory = path.resolve("src/app/api");
const hiddenApiDirectory = path.resolve(".pages-disabled-api");

if (existsSync(hiddenApiDirectory)) throw new Error(`Temporary directory already exists: ${hiddenApiDirectory}`);
renameSync(apiDirectory, hiddenApiDirectory);
try {
  rmSync(path.resolve(".next"), { recursive: true, force: true });
  const next = path.resolve("node_modules/next/dist/bin/next");
  const result = spawnSync(process.execPath, [next, "build", "--webpack"], { stdio: "inherit" });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exitCode = result.status ?? 1;
} finally {
  renameSync(hiddenApiDirectory, apiDirectory);
}