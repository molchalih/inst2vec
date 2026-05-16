import inspect


def test_default_cases_tuple():
    from modules.embeddings import DEFAULT_CASES

    assert DEFAULT_CASES == ("video", "sandwich", "audio")


def test_embed_clip_embeddings_signature():
    from modules.embeddings import embed_clip_embeddings

    sig = inspect.signature(embed_clip_embeddings)
    assert "settings" in sig.parameters
    assert "cases" in sig.parameters
    assert sig.parameters["cases"].default is None


def test_embed_user_embeddings_signature():
    from modules.embeddings import embed_user_embeddings

    sig = inspect.signature(embed_user_embeddings)
    assert "settings" in sig.parameters
    assert "cases" in sig.parameters
    assert sig.parameters["cases"].default is None
