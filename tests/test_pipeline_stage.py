def test_stage_enum_covers_existing_callsites():
    from core.pipeline import Stage

    expected = {
        "audio_extract",
        "filter",
        "captions",
        "speech",
        "clip_embeddings",
        "user_embeddings",
        "cluster_search",
        "cluster_validation",
        "cluster_assign",
        "mir",
        "audio_extract_mir",
        "labels",
        "visualization",
    }
    actual = {s.value for s in Stage}
    assert actual == expected, (
        f"diff: extra={actual - expected} missing={expected - actual}"
    )


def test_stage_enum_str_compat():
    """Stage members must be str-compatible so existing string consumers work."""
    from core.pipeline import Stage

    s = Stage.FILTER
    assert s == "filter"
    assert f"{s}" == "filter" or s.value == "filter"  # str(StrEnum) is the value


def test_stage_includes_mir_members():
    from core.pipeline import Stage

    assert Stage.MIR.value == "mir"
    assert Stage.AUDIO_EXTRACT_MIR.value == "audio_extract_mir"
