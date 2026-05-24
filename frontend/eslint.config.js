import js from "@eslint/js";
import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";
import importPlugin from "eslint-plugin-import";
import globals from "globals";

// Layer order: app → features → ui → interaction → render → state → data → core.
// `no-restricted-paths` zones below catch BOTH `@/...` and relative imports
// (../, ../../) because the rule resolves to the target file and compares
// against the directory globs.
const layerZones = [
  // core/ is framework-free and depends on nothing else in src/.
  { target: "./src/core", from: "./src/data" },
  { target: "./src/core", from: "./src/state" },
  { target: "./src/core", from: "./src/render" },
  { target: "./src/core", from: "./src/interaction" },
  { target: "./src/core", from: "./src/ui" },
  { target: "./src/core", from: "./src/features" },
  { target: "./src/core", from: "./src/app" },

  // data/ depends only on core.
  { target: "./src/data", from: "./src/state" },
  { target: "./src/data", from: "./src/render" },
  { target: "./src/data", from: "./src/interaction" },
  { target: "./src/data", from: "./src/ui" },
  { target: "./src/data", from: "./src/features" },
  { target: "./src/data", from: "./src/app" },

  // state/ may import data, core only.
  { target: "./src/state", from: "./src/render" },
  { target: "./src/state", from: "./src/interaction" },
  {
    target: "./src/state",
    from: "./src/ui",
    except: ["./tokens.ts"],
  },
  { target: "./src/state", from: "./src/features" },
  { target: "./src/state", from: "./src/app" },

  // render/ and interaction/ may import state, core.
  {
    target: "./src/render",
    from: "./src/ui",
    // Exception: Pixi draw routines import tokens directly
    // (Pixi cannot see CSS); the rest of ui/ stays off-limits.
    except: ["./tokens.ts"],
  },
  { target: "./src/render", from: "./src/features" },
  { target: "./src/render", from: "./src/app" },
  {
    target: "./src/interaction",
    from: "./src/ui",
    // Exception: hooks that translate input to world-space need pixel
    // tokens (e.g. hover hit radius). The rest of ui/ stays off-limits.
    except: ["./tokens.ts"],
  },
  { target: "./src/interaction", from: "./src/features" },
  { target: "./src/interaction", from: "./src/app" },

  // ui/ may import core only (per the architecture table).
  { target: "./src/ui", from: "./src/data" },
  { target: "./src/ui", from: "./src/state" },
  { target: "./src/ui", from: "./src/render" },
  { target: "./src/ui", from: "./src/interaction" },
  { target: "./src/ui", from: "./src/features" },
  { target: "./src/ui", from: "./src/app" },

  // features/ never import from app/.
  { target: "./src/features", from: "./src/app" },
];

const FEATURE_NAMES = ["versions", "selection", "inspector", "search", "hover-tooltip"];
const crossFeatureZones = FEATURE_NAMES.flatMap((src) =>
  FEATURE_NAMES
    .filter((dst) => dst !== src)
    .map((dst) => ({
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
      "import/resolver": {
        typescript: { project: "./tsconfig.json" },
      },
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "import/no-cycle": "error",
      "import/no-internal-modules": ["error", {
        // Forbid deep @/ imports except @/ui/tokens (Pixi draw routines
        // need the token map directly; CSS can't reach Pixi). Minimatch's
        // !(...) extglob misbehaves with the @/*/* shape, so layers are
        // enumerated explicitly.
        forbid: [
          "@/+(core|data|state|render|interaction|features|app)/*",
          "@/+(core|data|state|render|interaction|features|app)/**",
          "@/ui/!(tokens)",
          "@/ui/+(*)/**",
        ],
      }],
      "import/no-restricted-paths": ["error", {
        zones: [...layerZones, ...crossFeatureZones],
      }],
      "@typescript-eslint/no-unused-vars": ["error", { argsIgnorePattern: "^_" }],
    },
  },

  // Framework-free guard for core/: refuse react/pixi/jotai imports outright.
  {
    files: ["src/core/**/*.ts"],
    rules: {
      "no-restricted-imports": ["error", {
        paths: [
          { name: "react", message: "core/ is framework-free" },
          { name: "react-dom", message: "core/ is framework-free" },
          { name: "pixi.js", message: "core/ is framework-free" },
          { name: "@pixi/react", message: "core/ is framework-free" },
          { name: "jotai", message: "core/ is framework-free" },
        ],
      }],
    },
  },

  // .tsx allowed only in render/, ui/, features/*/ui/, app/. Elsewhere, lint errors via the file glob.
  {
    files: [
      "src/core/**/*.tsx",
      "src/data/**/*.tsx",
      "src/state/**/*.tsx",
      "src/interaction/**/*.tsx",
    ],
    rules: {
      "no-restricted-syntax": ["error", {
        selector: "Program",
        message: ".tsx files are not permitted in this layer; pure logic goes in .ts",
      }],
    },
  },
);
