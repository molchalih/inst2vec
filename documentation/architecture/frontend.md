# Architecture

This document is the architecture contract for the frontend atlas viewer. It
explains the layering, state, data, and rendering rules and why they exist.
Keep it tight — if a section grows beyond two screens, break it out.

## 1. Layering

The codebase is organised as a stack of layers. **Imports flow strictly
downward.** A higher layer may depend on lower layers; a lower layer
must never reach upward.

```
┌───────────────────────────────────────────────────────────┐
│ app/         composition root — wires everything          │
├───────────────────────────────────────────────────────────┤
│ features/    self-contained slices that may grow large    │
├───────────────────────────────────────────────────────────┤
│ ui/          DOM primitives, design tokens                │
├───────────────────────────────────────────────────────────┤
│ interaction/ React hooks bridging input → state           │
├───────────────────────────────────────────────────────────┤
│ render/      Pixi adapters and layers                     │
├───────────────────────────────────────────────────────────┤
│ state/       jotai atoms — one per concern                │
├───────────────────────────────────────────────────────────┤
│ data/        BulkSource + ApiClient, schemas              │
├───────────────────────────────────────────────────────────┤
│ core/        pure TS — geometry, palette, hit-test, math  │
└───────────────────────────────────────────────────────────┘
```

### Allowed import edges

| From            | May import from                                  |
| --------------- | ------------------------------------------------ |
| `app/`          | everything                                       |
| `features/X/`   | `ui`, `interaction`, `render`, `state`, `data`, `core` |
| `ui/`           | `core`                                           |
| `interaction/`  | `state`, `core`, plus `@/ui/tokens` only         |
| `render/`       | `state`, `core`, `interaction`, plus `@/ui/tokens` only |
| `state/`        | `data`, `core`                                   |
| `data/`         | `core`                                           |
| `core/`         | nothing inside `src/`                            |

`features/X/` never imports from `features/Y/`. Cross-feature concerns
are promoted down to `state/`, `data/`, or `core/`.

Three intentional carve-outs make the table read awkward but exist for
real reasons:

- **`@/ui/tokens` only.** `render/draw/` (Pixi cannot see CSS) and a
  small number of `interaction/` hooks (screen-pixel hit radii, motion
  durations) consume the token map directly. The rest of `ui/` stays
  off-limits. The lint config enforces this narrowly via an `except:
  ["./tokens.ts"]` clause; no other `ui/` modules can leak into lower
  layers.
- **`render/` may import from `interaction/`.** Motion hooks like
  `useTween` are framework-bound (React + rAF) and are reused by both
  `render/` layers and `interaction/` hooks. `core/` cannot host them
  (no React); duplicating them would split the source of truth. The
  one-way ordering of `interaction → render` survives because no
  `interaction/` module imports from `render/`.
- **DI singletons in `state/`.** `state/bulk-singleton.ts` exports a
  module-level `BulkSource` registrar (`setBulkSource`,
  `requireBulkSource`). It lives in `state/` rather than `data/`
  because its sole purpose is to let atoms reach side effects
  *without* React context — keeping `ensureRunAtom` & friends
  synchronous and free of provider plumbing. `state/api-singleton.ts`
  hosts the `ApiClient` registrar (`setApiClient`, `requireApiClient`)
  by the same pattern; atoms (`ensureClusterDetailAtom`,
  `ensureCreatorDetailAtom`) reach the API without React context.
  `state/` thus owns one tiny channel into `data/` that the atoms
  themselves cannot import (`state → data` is allowed; the singleton
  is the inversion point).

### Why one-way layers

The prior atlas codebase died of cyclic ownership: the canvas reached
into the store, the store reached into the search index, the search
index reached into selection, selection reached back into the canvas.
You couldn't change one without breaking three. A directed acyclic
import graph makes that mechanically impossible.

## 2. Public surface

Every folder has an `index.ts` that re-exports its public API. Anything
not re-exported is internal — consumers must not import it. This lets
us refactor internals freely without auditing every call site.

```ts
// state/index.ts
export { viewportAtom, useViewport } from "./viewport.atom";
export { hoverAtom, useHover } from "./hover.atom";
// ...
```

```ts
// ✅ allowed
import { useViewport } from "@/state";

// ❌ rejected by eslint-plugin-import (no-internal-modules)
import { viewportAtom } from "@/state/viewport.atom";
```

## 3. The `core/` layer

`core/` is the only layer that has a single absolute rule: **no
framework imports**. No `react`, no `pixi.js`, no `jotai`, no `window`,
no `document`. Plain TypeScript. The reasons:

- It is the layer most likely to be reused (CLI tools, server-side
  prerender, future native viewer).
- It is the layer where bugs cost the most, because everything depends
  on it.
- Pure modules are trivially testable, and we test them all.

Each module is a small, fully-tested file with one clear contract. The
current modules — `geom/` (vectors, transforms, fit/clamp math, ellipse +
stretch helpers), `motion/` (easing, the per-id hash, the intro
timeline/stagger), `morph/` (run-to-run interpolation for version
switches), `palette/`, `viewport/`, and `spatial/` (hit-test) — are
representative; `distinctiveness/` and `format/` (inspector-facing
text/number helpers) follow the same shape. The rule is the invariant,
not the list: pure TypeScript, one concern per file, a colocated
`*.test.ts`.

## 4. State (`state/`)

State lives in **multiple jotai atoms**, one concern per file. A typical
atom file is ~30 LOC and exports an atom plus a hook that reads/writes
it. Atoms compose; we use derived atoms for projections rather than
duplicating state.

One concern per atom file (~30 LOC), composed via derived atoms rather
than duplicated state. Representative atoms: `viewport.atom` (derived fit
+ optional override), `selection.atom` (a tagged union, not a bare id),
`run.atom` (a cache plus the `ensureRun` writer with its post-await
guard), and `transition.atom` (non-null only during a version switch).
Detail atoms (`cluster-detail`, `creator-detail`) share a
fetch-into-map shape keyed by id; `intro.atom` mirrors `transition.atom`
(a per-frame state atom + a driver atom + a one-shot `introPlayed` guard).
New atoms join by adding a file, never by widening an existing one.

### `selection.atom.ts` — discriminated union, not a bare id

`Selection` is a tagged union rather than `{ id?: number }` because the
*kind* of thing selected determines which camera-focus strategy and which
inspector pane applies. Collapsing kind into a single optional id would
force every consumer to infer the kind from context — a hidden coupling.

```ts
export type Selection =
  | { kind: "cluster"; clusterId: number }
  | { kind: "creator"; creatorId: number }
  | null;
```

`selectionAtom` holds the current selection; `selectDotAtom` is the
write-only action-atom that resolves a dot-id to the correct variant.
When the API is disabled, a dot click maps to `"cluster"` (the cluster
label is in the bulk payload). When the API is enabled, it maps to
`"creator"` (the detail panel pulls from `ApiClient.getCreator`). The
discrimination happens once, in `selectDotAtom`, so `Inspector` and
`useCameraFocus` branch on `kind` cleanly without re-doing the lookup.

### `visible-rect.atom.ts` — the usable canvas sub-rect

`visibleRectAtom` is derived from `selectionAtom` and `viewportSizeAtom`.
When selection is null it equals the full viewport; when a selection is
open it subtracts the inspector panel's width from the left edge. Every
consumer that needs to know "where can the camera point without being
hidden by chrome" reads this atom — not `viewportSizeAtom` directly.

The boundary is in `state/` rather than inside the `inspector` feature
because `useCameraFocus` (also in `inspector`) needs the rect to compute
focus targets, and `state/` is the right layer for atoms that two
in-feature concerns share. The panel width constant comes from
`ui/tokens` via the one documented `state → @/ui/tokens` carve-out.

Rules:

- A single atom file should rarely exceed ~50 LOC. Split if it grows.
- Atoms hold domain data, not view data. The `screenX/Y` in
  `hover.atom.ts` is the exception (it's where the cursor is) and is
  scoped tightly.
- No "god atom". The temptation to add one field "just here" is how
  monolithic stores grow. Add a new atom instead.
- **Atoms are synchronous.** Data fetching is a side effect, not an
  atom. The composition root (`app/`) owns the `useEffect` that calls
  `bulkSource.getRun(...).then(setRun)`. Atoms never `await`.

### `run.atom.ts` — cache, plus a derived active-run-id

`activeRunId` is **derived**: `state/active-run-id.atom.ts` reads
`caseAtom` + `manifestAtom.default_run_id` and returns the runId the
active case should load. The cache lives separately:

```ts
type RunState = { runs: Map<string, AtlasRun>; activeRunId: string | null };

export const runStateAtom = atom<RunState>({ runs: new Map(), activeRunId: null });
export const activeRunAtom = atom((get) => {
  const s = get(runStateAtom);
  return s.activeRunId ? s.runs.get(s.activeRunId) ?? null : null;
});
```

`ensureRunAtom` is the only writer. It (a) marks the requested runId on
`requestedRunIdAtom`, (b) awaits a fetch if the cache misses, (c)
re-checks `requestedRunIdAtom` before flipping `activeRunId` — so a
slow earlier fetch cannot clobber the activeRunId selected by a newer
request. The `runStateAtom.activeRunId` field is the *committed*
pointer; the derived `activeRunIdAtom` in `state/active-run-id.atom.ts`
is the *intended* one. `AppShell` wires them together: when the
intended id changes, an effect calls `ensureRunAtom`, which advances
the committed pointer once the run is in the cache. `bulkSource` is
provided via the module-level singleton in `state/bulk-singleton.ts`
(see §1 "DI singletons" carve-out), so the atom stays synchronous and
free of React context. Map updates are immutable; never mutate in
place.

### Coordinate spaces — raw vs. stretched

`users.json` and `clusters.json` ship raw UMAP coordinates. Two
visualisation problems follow from that: square UMAP layouts leave
huge vertical gutters in a 16:9 viewport, and cluster shapes carry
no visual budget for the chrome around the canvas.

We resolve both by deriving a `stretchedRunAtom` from `activeRunAtom`
and `viewportSizeAtom`: a one-time anisotropic scale per axis,
centered on the origin, that maps the run's bounds onto the viewport.
Layers, hit-tests, and `fitBounds` consume the stretched run.

The interactive transform (`viewportAtom`) keeps uniform scale, so
`Transform`, `fitBounds`, and `applyWheel` are unchanged. URL state
(`route.atom`) carries no world coordinates and stays raw.

### `viewportAtom` is derived, not stored

`viewportAtom` reads as a function of `(stretchedRun, viewportSize)`
with an optional override layer for pan/zoom and case-switch easing.
The first frame consumers see is already the correct fit, so there is
no "render at identity then snap" timing window — this matters because
`@pixi/react`'s `Application` captures children JSX on first mount and
re-applies it after an async init, so any stale first-mount state
would otherwise clobber later commits. `Stage` gates the Pixi root on
run-availability for the same reason.

The trade-off: cluster ellipse aspect ratios in the rendered frame
no longer match raw UMAP aspect ratios. We accept that: users see
the stretched frame only, and the cluster *grouping* (which dots
belong together) is preserved exactly.

The stretch helpers live in two layers:
- `core/geom/stretch.ts` — pure `stretchPoint`, `stretchEllipse` math
  (framework-free, fully tested).
- `data/transform.ts` — `stretchRun(run, width, height)` composes the
  helpers over `AtlasRun` to produce a viewport-aligned copy. Pure.

Writes to `viewportAtom` go through `core/viewport/clampPanZoom`,
which clamps scale to a configurable factor of the fit-scale
(`tokens.viewport.clamp.{min,max}ScaleFactor`) and clamps translation
so the content bounds always intersect the viewport (with `panMarginPx`
of allowed overrun). Both pan/zoom gestures and the case-switch ease
travel through this single chokepoint; `fitBounds` results are by
construction inside the clamp, so the initial fit is a no-op. The
canonical guardrail values live in `ui/tokens.ts` alongside every
other tunable; `state/` reads them via a narrowly-scoped lint
carve-out (`state/` → `@/ui/tokens` only), symmetric with the
existing `render/` and `interaction/` carve-outs.

## 5. Data — two planes, never merged

The frontend has **two independent data planes** with different
lifecycles, delivery models, and cache strategies. They are never
combined into a single interface, and the bulk payload is never widened
to carry detail data.

### 5a. Bulk plane (`data/bulk/`)

What the static bundle ships:

- `manifest.json` — list of runs (one per embedding case)
- per-run `users.json` — positional array `[id, x, y, cluster_id, has_detail, centrality]`
- per-run `clusters.json` — ellipse geometry + label

That is the entire bulk payload. Forever. It does not carry
biographies, avatars, follower edges, search indexes, or per-creator
metadata. Its size scales with the number of runs (3 today) and the
number of users (~5k → ~50k), not with the per-row column count.

The on-disk layout is **versioned by embedding case** from day one:

```
public/data/
  manifest.json
  runs/
    <run-id>/users.json          # bulk positions
    <run-id>/clusters.json       # bulk ellipse geometry
    <run-id>/users/<id>.json     # per-creator detail (StaticApiClient)
    <run-id>/clusters/<id>.json  # per-cluster detail (StaticApiClient)
```

Interface:

```ts
export interface BulkSource {
  getManifest(): Promise<Manifest>;
  getRun(runId: string): Promise<AtlasRun>;
}
```

v1 ships `StaticBulkSource` (fetch + Zod validate against committed
JSON). It is the only `BulkSource` we expect to need.

### 5b. API plane (`data/api/`)

Every read triggered by a user gesture goes through the API:

- click a dot → `api.getCreatorDetail(id)`
- type in search → `api.searchCreators(query)`
- open a creator card → `api.getReels(creatorId)`
- toggle follower graph → `api.getEdges(creatorId)`

These are network calls at interaction time. They are never
prefetched into the static bundle. This is the entire reason the bulk
payload stays small forever.

The `ApiClient` interface is the single Postgres-shaped contract:
`getClusterDetail(id)` and `getCreatorDetail(id)` (always answerable),
plus live-only `searchCreators`/`getEdges`/`getReels`. v1 ships
`StaticApiClient`, which serves cluster + creator detail from per-id
JSON shipped beside the bulk payload and throws `ApiUnavailableError`
for the live-only methods. `HttpApiClient` is the drop-in for the
FastAPI deploy — flip `config.apiBaseUrl` and no other code changes.
The client reads the active runId on every call (via a getter) so it
follows version switches without being recreated.

### 5c. Why this split is permanent

- **Cold-start performance.** The bulk payload determines time-to-dots
  on a CDN. Keeping it tight (`~3 numbers × N users + cluster
  geometries`) is the difference between sub-second first render and a
  spinner.
- **Cost.** Static hosting is free; an API has a request budget.
  Pre-bundling detail data would burn the bandwidth even when nobody
  clicks.
- **Freshness.** A creator's reels, follower count, or thumbnail can
  change daily. Static bundles deploy on cluster runs, not daily.
- **Privacy.** No PII ever lands in the bundle, by construction. The
  bulk payload literally cannot leak biographies or handles because it
  does not contain them.
- **Predictable growth.** When a new interaction lands (e.g. "show
  followers"), it adds an `ApiClient` method, not a bulk-payload
  field. The contract for what ships in the bundle is closed.

### 5d. Schemas

Schemas live in `data/schemas/` and are Zod. `data/schemas/version.ts`
exports `SCHEMA_VERSION` — the wire/contract version (currently 7) that
every payload is pinned to; mismatched payloads hard-fail at load. The
bulk payload's tuple shape is separately named `bulk-v2` (the v2 there
labels the bulk-tuple revision, not the wire version); detail payloads
have their own `cluster-detail`/`creator-detail` schemas, all pinned to
the same `SCHEMA_VERSION`. When the contract evolves, bump
`SCHEMA_VERSION` and add a migrator — never edit an existing schema
in-place.

### 5e. Caching

- **Bulk** — browser HTTP cache + Cache-Control on the deploy. We do
  not maintain an in-memory layer beyond what jotai already gives us
  (one fetched `AtlasRun` lives in an atom).
- **API** — TanStack Query is the obvious tool when `ApiClient` lands.
  Not added in v1 (no API to cache). Reserved name:
  `data/api/queryClient.ts`.

## 6. Render (`render/`)

Pixi is an implementation detail. The rules:

- One `<pixiGraphics>` per layer (dots, ellipses, hover overlay). Never
  one component per element.
- Layers subscribe to atoms via `useAtomValue` and redraw on change.
  React reconciliation is deliberately not in the hot path.
- Pixi never receives pointer events. The wrapper `<div>` does, and
  `interaction/useHover` translates to world coordinates.

### Layer/draw split

Each layer component is thin: it subscribes to the atoms it needs and
delegates the actual drawing to a pure function in `render/draw/`.

```
render/
  Stage.tsx            <Application> + container + drag-catcher
  frame.ts             DotsFrame / EllipsesFrame types + run→frame builders
  layers/
    DotsLayer.tsx      useAtomValue → useMemo(frame) → drawDots(g, frame, viewport)
    EllipsesLayer.tsx  ...
    HoverLayer.tsx     active hover overlay (one shape per concern)
    TrackingLayer.tsx  persistent tracked-dot beacon (halo + breathing border)
  draw/
    drawDots.ts        (g, frame, viewport) → void   pure
    drawEllipses.ts    (g, frame, viewport) → void   pure
    drawHoverDot.ts    (g, dotId, run, viewport, pulse) → void
    drawHoverCluster.ts(g, slots, run, viewport) → void
    drawTrackingHalo.ts   (g, pos, viewport, pulse, vis) → void
    drawTrackingMarker.ts (g, pos, viewport, pulse, vis) → void
```

Why the split: `draw/*` functions are framework-aware (they call
`graphics.circle(...)`) but UI-state-free. They take everything they
need as arguments, so they are testable with a mock `Graphics`
instance and reusable across layers (e.g. a future minimap reuses
`drawDots` with a different transform).

The `DotsFrame` / `EllipsesFrame` types — defined in `render/frame.ts`
— are the extension point. A frame bundles already-resolved drawables
(positions, colours, per-element alpha) plus a global `alphaScale` and
`radiusScale`/`strokeWidthScale`. The frame builder either reads the
stretched active run (idle) or interpolates between two runs (during
a version-switch transition). The draw functions never branch on
transition state — they consume whichever frame the layer hands them.

`DotsLayer` and `EllipsesLayer` carry an intro branch
(`runToIntroDotsFrame` + `introEllipseAlpha`), checked *after* the
transition branch. The intro is a deliberate, one-time animation kept
fully separate from the version-switch engine — they share only the
pure `core/motion` helpers (`introPhaseAndProgress`, `introEllipseAlpha`,
etc.). When no transition and no intro is active the layers fall through
to the static idle frame.

## 7. Interaction (`interaction/`)

Thin React hooks that bridge browser input to atoms:

- `usePanZoom()` — wheel + drag listeners; writes `viewportAtom` /
  `wheelZoomAtom`.
- `useHover()` — mousemove → world coords → `HitTest` → writes
  `hoverAtom`. Throttled to one rAF. Bails when `hitTestAtom` is null
  (no run yet, or a transition is in flight).
- `useClick()` — click → reads `hoverAtom` → dispatches `useSelectDot`
  on a dot, `useClearSelection` on empty canvas. The selection kind
  (cluster vs. creator) is chosen inside the state writer based on
  `isApiEnabled()`; the click hook stays a thin input bridge.
- `useUrlSync()` — bidirectional sync of `routeAtom` ↔
  `window.location.hash`.
- `useTrackViewportSize()` — `ResizeObserver` on the wrapper, writes
  `viewportSizeAtom`.
- `useFitOnActiveRun()` — clears the viewport override on first run
  and on resize so the derived fit takes over. Case-switch ease is
  **not** here — the transition engine owns it.
- `useEscKey(open, handler)` — generic Escape listener gated by an
  `open` flag.
- `useTween(target, duration, ease)` — generic scalar rAF tween.
- `useEasedScalar(active, duration, ease)` — boolean → 0/1 tween;
  thin wrapper over `useTween` for layer-anim ergonomics.
- `useCrossfadeSlots(activeId, duration, ease)` — A→B handoff helper
  for singleton overlay ids (used by `HoverLayer`'s cluster overlay).
- `useVersionTransition()` — observes the intended active runId; when
  it changes, drives the four-phase transition timeline (camera reset
  → cluster-out → dot-morph → cluster-in) and clears hover/hit-test
  state for the duration.
- `useIntroAnimation()` — drives the one-time page-load entrance;
  seeds once when the first run is ready and clears itself when done.

Hooks contain glue, not logic. The math is in `core/`; the data flow
is in `state/`. Anything else is wrong.

## 8. UI (`ui/`)

DOM primitives and design tokens. v1 ships `Tooltip` mounted; `Panel`
and `SearchInput` ship as working stubs (correct shape, not yet
composed by any feature).

```
ui/
  tokens.ts           single source of truth for visual constants
  primitives/
    Tooltip.tsx       positioned div, viewport-clamped
    Panel.tsx         slide-in panel shell (stub-ready)
    SearchInput.tsx   styled input + hotkey (stub-ready)
```

### Tokens are the single source of truth

`tokens.ts` exports a typed `const` object with colours, spacing,
radii, and motion durations. It is consumed by **three** places:

1. `tailwind.config.ts` — bridged via CSS variables so utility classes
   resolve to token values (`bg-bg-canvas`, `text-fg-muted`). Colour
   tokens are expressed in `rgb(var(--name) / <alpha-value>)` form so
   slash-opacity utilities (`bg-bg-canvas/95`) emit valid CSS.
2. DOM components in `ui/primitives/` and `features/*/ui/` — via
   Tailwind utilities.
3. Pixi draw routines in `render/draw/` — by direct import (Pixi
   doesn't see CSS).

Never hardcode a colour or spacing value in a component. If a value is
not in `tokens.ts`, add it there first.

### CSS lives in two places, on purpose

- `index.html` `<style>` — the `:root` custom-property definitions
  (`--bg-canvas`, `--fg-default`, `--fg-muted`) and the `html/body/#root`
  base layout. **Inlined in the document head because they must paint
  on the first frame, before the JS bundle loads.** Without this, the
  page flashes white before Tailwind utilities resolve. Values are
  expressed as space-separated RGB channels (e.g. `11 18 32`) so the
  Tailwind `rgb(... / <alpha-value>)` colour pattern works. A
  `<meta name="color-scheme" content="dark">` + `color-scheme: dark` on
  `:root` tells the user agent to use a dark canvas as its pre-CSS
  default, so the brief moment before our `<style>` is applied isn't
  white either.
- `src/styles.css` — only the three `@tailwind` directives. Anything
  else belongs in `tokens.ts` (consumed via Tailwind) or in a
  component-local class.

Values in `index.html` mirror `tokens.ts` literally; both files
deliberately carry the same colour. When changing a base colour, update
both and run the build (the inline block has no compile-time link to
`tokens.ts`).

## 9. Features (`features/`)

A **feature slice** is a folder owning one user-visible capability
end-to-end. It may contain its own `ui/`, hooks, a feature-local atom,
and an API wrapper. It must not reach into other features.

### Slice template

```
features/<name>/
  index.ts          public surface (usually one component)
  ui/               components (optional)
  hooks/            feature-local hooks (optional)
  state.atom.ts     feature-local atom (optional)
  api.ts            ApiClient method wrappers (optional)
```

### Slice rules

1. May import from `ui`, `interaction`, `render`, `state`, `data`,
   `core`. Never from another feature.
2. `index.ts` is the sole public surface. Internals are private.
3. A feature that needs API data consumes `ApiClient` from React
   context (provided by `app/`). If `apiEnabled === false`, the
   feature **unmounts itself at the top of its tree** — child
   components do not guard individually.
4. A v1 stub feature renders `null`. No placeholder UI, no "coming
   soon" copy, no empty modal. Invisible until it works.

The contract for each feature lives in this §9 table plus each
feature's `index.ts` public surface.

### v1 slices

Six slices mount unconditionally in `AppShell`. Four ship rendered UI
(`versions/`, `hover-tooltip/`, `inspector/`, `tracking/`); the rest stay
`null` stubs with real contracts.

| Slice            | v1 behaviour                                            |
| ---------------- | ------------------------------------------------------- |
| `versions/`      | Collapsible top-anchored drawer with one pill per embedding case; closed by default, opens via the chevron-tongue, Esc closes |
| `hover-tooltip/` | Resolves cluster label from `hoverAtom` + `activeRunAtom`; mounts `<Tooltip>` at the cursor |
| `selection/`     | Observes `selectionAtom`; renders `null`                |
| `inspector/`     | Selection-driven side panel. Engine: `selectionAtom` (read), `useCameraFocus` (camera ease on selection change), `visibleRectAtom` (sub-rect for focus math). Components: `ClusterPane` (bulk data) and `CreatorPane` (renders real creator detail via `StaticApiClient.getCreatorDetail`) compose a `PaneShell`/`Section*`/`Chip`/`AudioBar`/`MicroBar` primitive tree. `AppShell` also mounts a headless `<CameraFocus />` driver that runs `useCameraFocus`. |
| `search/`        | Registers `Cmd/Ctrl+K`; opens a sheet that renders `null` (depends on `ApiClient.searchCreators`) |
| `tracking/`      | Point-tracking toggle. Renders `TrackingControl`, a glyph button in the `ControlDock` that flips tracking mode (toggling off clears the tracked creator), disabled-faded during the intro and version-switch transitions. When a creator is tracked, `TrackingLayer` (in `render/`) draws a persistent beacon — soft halo + breathing border ring — on that dot, which `AppShell`'s `TrackingPrefetch` keeps glued across version switches by warming every run into the cache |

Future work fills the stubs in without touching `app/` or `render/`. The
features expand inside their own folder; the composition root never
changes shape.

## 10. App (`app/`)

The composition root. The only layer that imports from everything.

```
app/
  AppShell.tsx     headless effects + the rendered tree
  providers.tsx    Providers: JotaiProvider + ApiRegistration; setBulkSource + setApiClient
  config.ts        typed env config + feature flags
  routes.ts        URL state schema (Zod), consumed by route.atom.ts
  index.ts
```

`src/main.tsx` is the React root entry; it mounts `<AppShell />`. The
file lives at `src/main.tsx`, not `app/main.tsx`, because Vite resolves
the entry from `index.html`'s script tag and we keep that path stable.

### Dependency injection

`providers.tsx` exposes `Providers`, which mounts `JotaiProvider` then
`ApiRegistration`. It constructs the `StaticBulkSource` and registers it
via `setBulkSource`, and constructs the `ApiClient` (`HttpApiClient` when
`config.apiBaseUrl` is set, else `StaticApiClient` reading the active
runId through a store getter) and registers it via `setApiClient`. Both
are also provided through React context (`useBulk`, `useApi`) so tests can
wrap a tree in a mock without touching production code.

### Composition is static, behaviour is dynamic

`AppShell` mounts every feature unconditionally. Whether a feature
renders something is the feature's decision — never the composition
root's. The tree shape stays stable across the foundation's lifetime;
features evolve inside their own folders.

```tsx
<Providers>
  <Routing />          {/* useUrlSync */}
  <ViewportTracker />  {/* useTrackViewportSize */}
  <RunLoader />        {/* ensureManifestAtom + ensureRunAtom */}
  <TrackingPrefetch /> {/* warms every run into the cache while a creator is tracked */}
  <Fitting />          {/* useFitOnActiveRun */}
  <Intro />            {/* useIntroAnimation */}
  <VersionTransition />{/* useVersionTransition */}
  <CameraFocus />      {/* useCameraFocus — eases viewport on selection change */}
  <main>
    <Stage>
      <EllipsesLayer />
      <DotsLayer />
      <TrackingLayer />
      <HoverLayer />
    </Stage>
    <VersionsFeature />
    <HoverTooltipFeature />
    <SelectionFeature />
    <InspectorFeature />
    <SearchFeature />
    <ControlDock>
      <TrackingFeature />
    </ControlDock>
  </main>
</Providers>
```

### Headless effect components

Each of `Routing`, `ViewportTracker`, `RunLoader`, `TrackingPrefetch`,
`Fitting`, `Intro`, and `VersionTransition` is a component that returns
`null` and runs one hook (or effect). The pattern keeps `AppShell`
declarative — adding a new
app-wide effect is a 1-line mount, not an edit to a growing
`useEffect` inside `AppShell`. The feature components below `<main>`
are mounted unconditionally; each decides whether to render anything
based on its own state.

### URL-state schema

`state/route.schema.ts` is the single source of truth for what lives in the
hash (`routes.ts` re-exports it):

```ts
export const routeSchema = z.object({
  case: embeddingCaseSchema.optional(),
  cluster: z.coerce.number().int().optional(),
  user: z.coerce.number().int().optional(),
});
export type Route = z.infer<typeof routeSchema>;
```

`route.atom.ts` imports this schema to parse/serialize. New URL
parameters are added here first; features bind to them via the atom,
never by reading `window.location` directly.

## 11. Testing

- `core/` — every module has a colocated `*.test.ts`. Vitest. This is
  the test pyramid's base and it carries the load.
- `data/` — schema round-trips and source contract tests.
- `state/` — atom integration tests where derivations are non-trivial.
- `render/`, `interaction/`, `ui/`, `features/` — no automated tests in
  v1. Manual smoke + `tsc --noEmit` + successful `vite build` is the
  gate. Add tests when an area stops changing.

## 12. Out of scope

Things deliberately not in v1 but designed-for:

- LLM/manual cluster labels (export-only change; frontend reads `label`)
- Quadtree hit test (drop-in for `core/spatial`)
- Search index build + UI
- Admin tooling (kept out of the public bundle entirely — separate
  build target when it lands)
- Performance mode (DPR cap)

When any of these ships, it must respect the layer rules. If a feature
*can't* be built without breaking them, the rules win — file an issue
and discuss before bending them.

## 13. Designed-for, not built

The architecture deliberately leaves room for three near-future
expansions. Each gets a sentence so the next contributor doesn't
paint a corner.

### Edges (mutual subscriptions)

A new `render/layers/EdgesLayer.tsx` will sit between `EllipsesLayer`
and `DotsLayer` (`zIndex` 0.5). It consumes `state/edges.atom.ts`
(`Map<creatorId, number[]>`, sparse, keyed on selection) and draws
via `render/draw/drawEdges.ts`. The fetch goes through
`ApiClient.getEdges(creatorId)`; never bundled. Bundling math
(force-directed edge bundling) lives in `core/graph/`.

### Per-dot filter mask

`DotsFrame.users[i].alpha` is **already** the per-creator alpha
channel. A future filter pane (tag / follower threshold) computes a
mask and the frame builder multiplies it into each user's alpha. The
draw path needs no changes.

### Palette mode

`colorForCluster(clusterId)` becomes one of several palette functions
selected by a `colorMode` atom (cluster / follower-heatmap /
recency). The frame builder reads the active palette function from
state and resolves `user.color` accordingly. The draw path stays
identical.

### Inspector content kinds

`Selection` is a two-variant union today (`"cluster"` | `"creator"`).
Additional selection kinds — e.g. a saved search result, a follower-graph
focal node, an admin annotation — are additive: add a new variant to the
union, add a pane component, wire `selectDotAtom` to produce it, and
`Inspector` gains a new `else if` branch. The `Panel` chrome, the
`useCameraFocus` engine, and `visibleRectAtom` need no changes because
they branch on the presence/absence of a selection, not on its kind.

### Visible-rect inset stack

`visibleRectAtom` currently subtracts only the inspector panel width.
When additional chrome arrives (a bottom search bar, a top filter strip,
a minimap corner), the atom becomes a stack of insets: each piece of
chrome registers its own `Rect` inset and the atom computes the
intersection. The current single-panel derivation is a degenerate case
of that stack. Because `useCameraFocus` and `fitBoundsToRect` already
consume `visibleRectAtom` — not the raw viewport size — no camera-focus
or fit logic needs to change when new chrome arrives.

None of these is implemented in v1. They are documented here so
adding them later does **not** require rearchitecting the layer table
or the draw signatures.
