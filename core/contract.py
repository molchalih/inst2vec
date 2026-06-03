"""Single source of truth for the visualization JSON contract version.

Mirror of ``frontend/src/data/schemas/version.ts``. Bump when the JSON shape the
visualization export writes changes in a way the frontend Zod schemas would
reject; bumping forces every JSON file to be rewritten on the next run.

Lives in ``core/`` because it is a cross-cutting contract constant consumed by
the pipeline (``modules.visualization``), the serving decompose
(``core.database``), and the read API (``services.atlas_api``) alike — so no
layer has to reach into another to read it.
"""

SCHEMA_VERSION = 7
