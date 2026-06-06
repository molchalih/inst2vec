"""Tunable knobs for the Phase-2/3 backbone (design §5).

Every value is overridable via ``SA_<UPPER_FIELD>`` env vars. Defaults are the
design's proposed starting points; all are Phase-3 calibratable (plan §12).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Settings:
    # retirement / overlap
    min_overlap: int = 5
    max_overlap: int = 9
    confidence_threshold: float = 0.80
    # reliability / warm-up
    warmup_k: int = 8
    beta_alpha0: float = 2.0
    beta_beta0: float = 2.0
    dirichlet_conc: float = 1.0
    gold_blend_floor: float = 0.4
    # gold injection
    p_gold: float = 0.125
    gold_exposure_cap: int = 10  # max DISTINCT gold items per annotator
    # assignment lifecycle
    max_inflight: int = 2
    assign_ttl_s: int = 600
    sweep_interval_s: int = 300
    # balancer weights (w_info stays 0 until Phase 4 populates information)
    w_cover: float = 0.40
    w_bound: float = 0.25
    w_novel: float = 0.20
    eps_random: float = 0.05
    w_info: float = 0.0
    softmax_temperature: float = 0.5

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Load settings from environment variables with SA_<UPPER_FIELD> convention.

        Each field can be overridden via an env var named SA_ followed by the
        field name in uppercase (e.g., SA_MAX_INFLIGHT, SA_CONFIDENCE_THRESHOLD).
        Numeric values are automatically coerced to int or float based on the
        field's declared type.
        """
        env = os.environ if env is None else env
        kwargs: dict[str, object] = {}
        converters: dict[str, type] = {"int": int, "float": float}
        for f in fields(cls):
            key = f"SA_{f.name.upper()}"
            if key not in env:
                continue
            converter = converters.get(f.type)
            if converter is None:
                raise TypeError(f"Settings.{f.name} has unsupported type {f.type!r}")
            try:
                kwargs[f.name] = converter(env[key])
            except ValueError as exc:
                raise ValueError(f"invalid value for {key}={env[key]!r}: {exc}") from exc
        return cls(**kwargs)  # type: ignore[arg-type]
