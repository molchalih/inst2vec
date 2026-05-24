"""Storage abstraction tests using moto's in-process S3 mock."""

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")
from moto import mock_aws  # noqa: E402

from core.config import StorageSettings  # noqa: E402
from core.storage import ObjectStore  # noqa: E402


@pytest.fixture
def store(tmp_path):
    with mock_aws():
        # moto auto-creates an in-process S3 server scoped to the with-block.
        s = ObjectStore(
            settings=StorageSettings(
                backend="s3",
                bucket="test-bucket",
                prefix="videos/",
            ),
            endpoint_url=None,  # use moto default (real AWS endpoint stub)
            access_key="test",
            secret_key="test",
        )
        s.client.create_bucket(Bucket="test-bucket")
        yield s


def test_head_returns_none_when_object_missing(store):
    assert store.head("videos/12345.mp4") is None


def test_put_then_head_returns_size(store, tmp_path):
    p = tmp_path / "12345.mp4"
    p.write_bytes(b"\x00" * 4096)
    store.put(str(p), "videos/12345.mp4")
    meta = store.head("videos/12345.mp4")
    assert meta is not None
    assert meta["size"] == 4096


def test_put_is_idempotent(store, tmp_path):
    p = tmp_path / "x.mp4"
    p.write_bytes(b"abc")
    store.put(str(p), "videos/x.mp4")
    store.put(str(p), "videos/x.mp4")  # should not raise
    assert store.head("videos/x.mp4")["size"] == 3


def test_key_for_clip_uses_prefix(store):
    assert store.key_for_clip(12345) == "videos/12345.mp4"


def test_region_from_settings_is_passed_to_client():
    # RunPod's S3 endpoint signs with SigV4 against the datacenter region; an
    # unset/mismatched region yields SignatureDoesNotMatch.
    with mock_aws():
        s = ObjectStore(
            settings=StorageSettings(
                backend="s3", bucket="b", prefix="v", region="EU-RO-1"
            ),
            endpoint_url=None,
            access_key="t",
            secret_key="t",
        )
        assert s.client.meta.region_name == "EU-RO-1"


def test_key_for_clip_without_trailing_slash():
    s = ObjectStore(
        settings=StorageSettings(backend="s3", bucket="b", prefix="videos"),
        endpoint_url=None,
        access_key="t",
        secret_key="t",
    )
    assert s.key_for_clip(7) == "videos/7.mp4"
