import os
import threading
import time

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


class _ConcurrencyTrackingStore(_FakeStore):
    """Empty bucket (every clip is absent -> needs PUT). Each put() holds a slot
    briefly so overlapping uploads are observable, recording the peak count."""

    def __init__(self):
        super().__init__({})
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.puts = []

    def put(self, local_path, key):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self._lock:
            self.active -= 1
            self.puts.append(key)


def test_run_uploads_throttles_puts_independently_of_verify_workers(tmp_path):
    """PUTs must stay capped at put_workers even when the verify pool is wide,
    so a cold bucket can't start a verify-width storm of concurrent uploads."""
    store = _ConcurrencyTrackingStore()
    clips = []
    for i in range(1, 21):
        _write(tmp_path, f"{i}.mp4", 10)
        clips.append((i, str(tmp_path / f"{i}.mp4")))

    def key_for(cid):
        return f"videos/{cid}.mp4"

    results = run_uploads(store, clips, key_for, workers=20, put_workers=3)

    assert all(results.values())  # all uploaded
    assert len(store.puts) == 20
    assert store.max_active <= 3  # PUTs throttled despite 20 verify workers
