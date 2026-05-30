"""Read-only FastAPI service over the serving database.

Reconstructs the version-6 frontend contract (manifest + per-run users/clusters
+ per-id creator/cluster details) from the normalised ``serving_*`` tables and
serves it byte-identically to what ``modules.visualization.export`` writes to
disk. Reads ONLY the serving DB — never the pipeline main or identity DBs.

``build_app`` lives in ``services.atlas_api.app`` (imports FastAPI, an optional
dependency from the ``embedder`` group); ``reconstruct`` / ``serialize`` are
dependency-free so they can be imported without FastAPI installed.
"""
