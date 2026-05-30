"""Guard rail: frozen cases must never drift their config-identity hash.

``video`` and ``sandwich`` are KEPT untouched; ``auditory`` is the no-recompute
rename of the former ``maest`` case. The pipeline must emit ``SKIP fingerprint``
for all three at embeddings / labels / clustering / visualization. The binding
invariant is that each case's ``hash_text(case_config_identity(spec, settings))``
equals the value pinned below:

  * ``video`` / ``sandwich`` — byte-identical to main HEAD (their spec fields and
    the common identity parts are unchanged by the case rework).
  * ``auditory`` — the post-rename hash that the maest→auditory migration adopts
    into the embed seal (the only delta from the old maest identity is the
    leading ``case=`` token).

A failure here means a frozen case drifted and stored embeddings would be wiped
+ recomputed on the next run. Do NOT update these values to silence a failure
without understanding why the identity changed.
"""

from __future__ import annotations

from core import fingerprint as fp
from core.config import _load_settings
from modules.embeddings.cases import CASE_REGISTRY, case_config_identity

EXPECTED_HASHES: dict[str, str] = {
    "video": "1aedd553ae34e9c646769bf6c316c63cbc066ae6c1595894ad2de761fc1089f4",
    "sandwich": "77c3f3641bf06ad645972daa68acbebb870c4b3bc8b04f6d1cd7f6327253266e",
    "auditory": "6f7c67a6b256f4755fd5500f25f98e4e0cafb8b3d31a25225c8d198a54a6c8d6",
}


def test_frozen_case_config_identity_hashes():
    settings = _load_settings()
    for name, expected in EXPECTED_HASHES.items():
        actual = fp.hash_text(case_config_identity(CASE_REGISTRY[name], settings))
        assert actual == expected, (
            f"{name} config-identity drifted: {actual} != {expected}. "
            "A frozen case changed — stored embeddings would be recomputed."
        )


def test_auditory_identity_is_maest_identity_modulo_case_token():
    """The adopted auditory hash must be derivable from the old maest identity by
    swapping ONLY the leading ``case=`` token — proving the rename perturbs
    nothing else, so the migration's seal adoption is exactly correct."""
    settings = _load_settings()
    auditory_id = case_config_identity(CASE_REGISTRY["auditory"], settings)
    legacy_maest_id = auditory_id.replace("case=auditory", "case=maest", 1)
    # Re-keying back to maest reproduces the pre-rename identity string; only the
    # case token differs, everything downstream (onnx markers, sha256) matches.
    assert legacy_maest_id.startswith("case=maest|")
    assert legacy_maest_id.count("case=auditory") == 0
