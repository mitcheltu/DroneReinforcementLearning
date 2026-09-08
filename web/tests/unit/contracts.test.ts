import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { CONFIG_NAMES, ContractError, loadBytes, parseJson, validateArtifact,
  validateConfigBundle, verifyManifestConfigs, type ConfigBundle } from "../../src/contracts/loader";
import type { SchemaName } from "../../src/contracts/schemas";
import type { StateV1 } from "../../src/contracts/types";

const shared = new URL("../../../shared/", import.meta.url);
const cases = JSON.parse(readFileSync(new URL("fixtures/contracts/cases.json", shared), "utf8")) as {
  id: string; schema: SchemaName; data: unknown; accepted: boolean;
}[];

describe("shared acceptance corpus", () => {
  for (const test of cases) {
    it(test.id, () => {
      const original = structuredClone(test.data);
      if (test.accepted) expect(() => validateArtifact(test.schema, test.data)).not.toThrow();
      else expect(() => validateArtifact(test.schema, test.data)).toThrow(ContractError);
      expect(test.data).toEqual(original);
    });
  }
});

it("hashes original bytes, and checks expected hash", async () => {
  const raw = readFileSync(new URL("vehicle.v1.json", shared));
  const original = await loadBytes("vehicle", raw);
  const variant = Buffer.concat([raw, Buffer.from("\n ")]);
  const loaded = await loadBytes("vehicle", variant);
  expect(loaded.data).toEqual(original.data);
  expect(loaded.sha256).not.toEqual(original.sha256);
  await expect(loadBytes("vehicle", variant, original.sha256)).rejects.toThrow("SHA-256");
});

it.each([
  '{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}',
  '{"a":9007199254740992}', '{"a":1,}', '{/*comment*/"a":1}', '\ufeff{}',
])("rejects nonportable JSON %s", source => {
  expect(() => parseJson(new TextEncoder().encode(source))).toThrow(ContractError);
});

it("rejects invalid UTF-8", () => {
  expect(() => parseJson(new Uint8Array([255]))).toThrow(ContractError);
});

it("rejects nonfinite in-memory values", () => {
  for (const bad of [NaN, Infinity, -Infinity]) {
    const state = structuredClone(cases.find(c => c.id === "state-valid")!.data) as StateV1;
    state.position_m[0] = bad;
    expect(() => validateArtifact("state", state)).toThrow(ContractError);
  }
});

it("normalizes quaternion roundoff without mutating input", () => {
  const state = structuredClone(cases.find(c => c.id === "state-valid")!.data) as StateV1;
  state.quaternion_wxyz = [1.0000005, 0, 0, 0];
  const result = validateArtifact<StateV1>("state", state);
  expect(result.quaternion_wxyz).toEqual([1, 0, 0, 0]);
  expect(state.quaternion_wxyz[0]).toBe(1.0000005);
});

it("keeps schema resolution offline", () => {
  expect(fileURLToPath(shared)).toContain("shared");
  expect(() => validateArtifact("unknown" as SchemaName, {})).toThrow(ContractError);
});

it("rejects incompatible configurations and mismatched manifest bindings", async () => {
  const bundle = Object.fromEntries(await Promise.all(CONFIG_NAMES.map(async name =>
    [name, await loadBytes(name, readFileSync(new URL(`${name}.v1.json`, shared)))],
  ))) as ConfigBundle;
  validateConfigBundle(bundle);
  const manifest = {
    vehicle_sha256: bundle.vehicle.sha256,
    observation_sha256: bundle.observation.sha256,
    rules_sha256: bundle["course-rules"].sha256,
    training_config_sha256: bundle.training.sha256,
  };
  expect(() => verifyManifestConfigs(manifest, bundle)).not.toThrow();
  manifest.vehicle_sha256 = "0".repeat(64);
  expect(() => verifyManifestConfigs(manifest, bundle)).toThrow(ContractError);
  bundle.vehicle.data.rotor_max_thrust_n = 5;
  expect(() => validateConfigBundle(bundle)).toThrow(ContractError);
});
