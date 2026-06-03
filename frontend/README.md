# frontend

Static viewer for inst2vec clustering, deployed to GitHub Pages.

## Stack

Bun · Vite · React 19 · TypeScript · Pixi v8 · jotai · Tailwind.

## Local development

```bash
bun install
bun run dev
```

The dev server reads data from the read-only **atlas API** (the serving DB)
when `VITE_API_BASE_URL` is set, or from a locally generated `public/data/`
tree when it is unset. `public/data/` is **not** committed — generate it with
`scripts/publish_visualization.py` for an offline static run.

### Data source: HTTP API (default) or static JSON

The published Pages site points at the live atlas API by default
(`VITE_API_BASE_URL` is set in the Pages build). The API mirrors the static
JSON paths 1:1, so a static fallback is possible by unsetting
`VITE_API_BASE_URL` and committing a generated `public/data/` tree — but the
deploy no longer ships one.

## Quality gates

```bash
bun run typecheck   # tsc --noEmit
bun run lint        # eslint
bun run test        # vitest (all src/**/*.test.ts; coverage scoped to core/)
bun run build       # vite build
```

CI runs all four on every push and PR. A separate Pages workflow deploys the
build artefact to GitHub Pages after CI passes on `main`.

## Layout

The codebase is layered: `core` → `data` → `state` → `render` →
`interaction` → `ui` → `features` → `app`. Imports flow strictly
downward. See [`../documentation/architecture/frontend.md`](../documentation/architecture/frontend.md).
