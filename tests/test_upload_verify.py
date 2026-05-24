import os

from modules.upload import process_clip_upload, run_uploads, verify_outcome


class _FakeStore:
    def __init__(self, objects):
        self.objects = dict(objects)  # key -> size

    def head(self, key):
        if key in self.objects:
            return {"size": self.objects[key], "etag": '"x"'}
        return None


def test_absent_object_needs_upload():
    store = _FakeStore({})
    assert verify_outcome(store, "videos/1.mp4", local_size=100) == "absent"


def test_present_same_size_is_ok():
    store = _FakeStore({"videos/1.mp4": 100})
    assert verify_outcome(store, "videos/1.mp4", local_size=100) == "present"


def test_present_size_mismatch_needs_reupload():
    store = _FakeStore({"videos/1.mp4": 99})
    assert verify_outcome(store, "videos/1.mp4", local_size=100) == "mismatch"


class _RecordingStore(_FakeStore):
    def __init__(self, objects):
        super().__init__(objects)
        self.puts = []

    def put(self, local_path, key):
        self.puts.append(key)
        self.objects[key] = os.path.getsize(local_path)


def _write(tmp_path, name, size):
    p = tmp_path / name
    p.write_bytes(b"x" * size)
    return str(p)


def test_process_uploads_absent(tmp_path):
    store = _RecordingStore({})
    path = _write(tmp_path, "1.mp4", 10)
    assert process_clip_upload(store, "videos/1.mp4", path) == "uploaded"
    assert store.puts == ["videos/1.mp4"]


def test_process_skips_present(tmp_path):
    store = _RecordingStore({"videos/1.mp4": 10})
    path = _write(tmp_path, "1.mp4", 10)
    assert process_clip_upload(store, "videos/1.mp4", path) == "ok"
    assert store.puts == []


def test_process_reuploads_mismatch(tmp_path):
    store = _RecordingStore({"videos/1.mp4": 9})
    path = _write(tmp_path, "1.mp4", 10)
    assert process_clip_upload(store, "videos/1.mp4", path) == "uploaded"
    assert store.puts == ["videos/1.mp4"]


def test_process_reports_missing_local(tmp_path):
    store = _RecordingStore({})
    assert (
        process_clip_upload(store, "videos/1.mp4", str(tmp_path / "nope.mp4"))
        == "missing"
    )
    assert store.puts == []


def test_process_reports_failed_put(tmp_path):
    class _Boom(_RecordingStore):
        def put(self, local_path, key):
            raise RuntimeError("boom")

    store = _Boom({})
    path = _write(tmp_path, "1.mp4", 10)
    assert process_clip_upload(store, "videos/1.mp4", path) == "failed"


def test_run_uploads_sets_flags_from_real_state(tmp_path):
    # clip 1 already in bucket (size match), clip 2 absent, clip 3 missing local
    store = _RecordingStore({"videos/1.mp4": 10})
    _write(tmp_path, "1.mp4", 10)
    _write(tmp_path, "2.mp4", 10)

    clips = [
        (1, str(tmp_path / "1.mp4")),
        (2, str(tmp_path / "2.mp4")),
        (3, str(tmp_path / "3.mp4")),
    ]

    def key_for(cid):
        return f"videos/{cid}.mp4"

    results = run_uploads(store, clips, key_for, workers=4)

    assert results[1] is True  # present
    assert results[2] is True  # uploaded
    assert results[3] is False  # missing local
    assert set(store.puts) == {"videos/2.mp4"}
