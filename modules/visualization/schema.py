"""Single source of truth for the visualization JSON contract version.

Mirror of frontend/src/data/schemas/version.ts. Bump when the JSON shape
written by export.py changes in a way the frontend Zod schemas would
reject. Bumping it forces every JSON file to be rewritten on the next
pipeline run.
"""

SCHEMA_VERSION = 1
