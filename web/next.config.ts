import path from "node:path";
import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  // The simulator imports its shared JSON contracts from the repository-level
  // `shared/` directory. This is required for both dev and production builds.
  experimental: {
    externalDir: true,
  },
  turbopack: {
    root: path.resolve(__dirname, ".."),
  },
};

export default config;
