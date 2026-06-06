import js from "@eslint/js";
import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";
import importPlugin from "eslint-plugin-import";
import globals from "globals";

// Layer order: app → features → ui → interaction → state → data → core.
// (No render/ layer — this app is DOM + <video>, not Pixi.) `no-restricted-paths`
// zones catch both `@/...` and relative imports.
const layerZones = [
  // core/ is framework-free and depends on nothing else in src/.
  { target: "./src/core", from: "./src/data" },
  { target: "./src/core", from: "./src/state" },
  { target: "./src/core", from: "./src/interaction" },
  { target: "./src/core", from: "./src/ui" },
  { target: "./src/core", from: "./src/features" },
  { target: "./src/core", from: "./src/app" },

  // data/ depends only on core.
  { target: "./src/data", from: "./src/state" },
  { target: "./src/data", from: "./src/interaction" },
  { target: "./src/data", from: "./src/ui" },
  { target: "./src/data", from: "./src/features" },
  { target: "./src/data", from: "./src/app" },

  // state/ may import data, core only.
  { target: "./src/state", from: "./src/interaction" },
  { target: "./src/state", from: "./src/ui", except: ["./tokens.ts"] },
  { target: "./src/state", from: "./src/features" },
  { target: "./src/state", from: "./src/app" },

  // interaction/ may import state, core, plus @/ui/tokens only.
  { target: "./src/interaction", from: "./src/data" },
  { target: "./src/interaction", from: "./src/ui", except: ["./tokens.ts"] },
  { target: "./src/interaction", from: "./src/features" },
  { target: "./src/interaction", from: "./src/app" },

  // ui/ may import core only.
  { target: "./src/ui", from: "./src/data" },
  { target: "./src/ui", from: "./src/state" },
  { target: "./src/ui", from: "./src/interaction" },
  { target: "./src/ui", from: "./src/features" },
  { target: "./src/ui", from: "./src/app" },

  // features/ never import from app/.
  { target: "./src/features", from: "./src/app" },
];

const FEATURE_NAMES = ["judge", "onboarding", "session"];
const crossFeatureZones = FEATURE_NAMES.flatMap((src) =>
  FEATURE_NAMES.filter((dst) => dst !== src).map((dst) => ({
    target: `./src/features/${src}`,
    from: `./src/features/${dst}`,
  })),
);

export default tseslint.config(
  { ignores: ["dist", "node_modules", "coverage"] },
  {
    files: ["src/**/*.{ts,tsx}"],
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
      parserOptions: { project: "./tsconfig.json" },
    },
    plugins: { "react-hooks": reactHooks, import: importPlugin },
    settings: {
      "import/resolver": { typescript: { project: "./tsconfig.json" } },
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "import/no-cycle": "error",
      "import/no-internal-modules": [
        "error",
        {
          forbid: [
            "@/+(core|data|state|interaction|features|app)/*",
            "@/+(core|data|state|interaction|features|app)/**",
            "@/ui/!(tokens)",
            "@/ui/+(*)/**",
          ],
        },
      ],
      "import/no-restricted-paths": [
        "error",
        { zones: [...layerZones, ...crossFeatureZones] },
      ],
      "@typescript-eslint/no-unused-vars": ["error", { argsIgnorePattern: "^_" }],
    },
  },

  // Framework-free guard for core/.
  {
    files: ["src/core/**/*.ts"],
    rules: {
      "no-restricted-imports": [
        "error",
        {
          paths: [
            { name: "react", message: "core/ is framework-free" },
            { name: "react-dom", message: "core/ is framework-free" },
            { name: "jotai", message: "core/ is framework-free" },
          ],
        },
      ],
    },
  },

  // .tsx allowed only in ui/, features/, app/. Pure layers stay .ts.
  {
    files: [
      "src/core/**/*.tsx",
      "src/data/**/*.tsx",
      "src/state/**/*.tsx",
      "src/interaction/**/*.tsx",
    ],
    rules: {
      "no-restricted-syntax": [
        "error",
        {
          selector: "Program",
          message: ".tsx files are not permitted in this layer; pure logic goes in .ts",
        },
      ],
    },
  },
);
