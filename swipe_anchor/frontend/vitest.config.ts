import { defineConfig } from "vitest/config";
import { fileURLToPath, URL } from "node:url";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "happy-dom",
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
    setupFiles: ["./src/test-setup.ts"],
    server: { deps: { inline: ["zod"] } },
    coverage: {
      provider: "istanbul",
      include: ["src/core/**/*.ts"],
      exclude: ["src/core/**/*.test.ts", "src/core/**/index.ts"],
      thresholds: { lines: 100, branches: 100, functions: 100, statements: 100 },
    },
  },
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
});
