import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { expect, it } from "vitest";
import { CONFIG_NAMES, loadBytes } from "../../src/contracts/loader";

it("Python and browser-compatible loaders agree on full configurations and byte hashes", async () => {
  const root = fileURLToPath(new URL("../../../", import.meta.url));
  const python = process.env.AERORL_PYTHON ?? resolve(root,
    process.platform === "win32" ? ".venv/Scripts/python.exe" : ".venv/bin/python");
  expect(existsSync(python), "Set up the Python environment first").toBe(true);
  const reference = JSON.parse(execFileSync(python,
    ["-m", "training.scripts.check_contracts", "--shared", resolve(root, "shared")],
    { cwd: root, encoding: "utf8", timeout: 60000 }));
  for (const name of CONFIG_NAMES) {
    const actual = await loadBytes(name, readFileSync(resolve(root, "shared", `${name}.v1.json`)));
    expect(actual).toEqual(reference[name]);
  }
}, 60000);
