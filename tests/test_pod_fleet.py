import time

from core.runpod import RunPodClient
from modules.embeddings.fleet import (
    PodFleet,
    fleet_enabled,
    read_reconcile,
    write_reconcile,
)


class _FakeBackend:
    def __init__(self):
        self.api_key = None
        self.created = []
        self.terminated = []

    def create_pod(self, **kwargs):
        self.created.append(kwargs)
        return {"id": f"pod{len(self.created)}", "name": kwargs["name"]}

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)

    def get_pods(self):
        return []


def _spec_kwargs():
    from core.runpod import PodSpec

    return PodSpec(
        image="img",
        gpu_type_ids=("g",),
        data_center_id="dc",
        network_volume_id="v",
        volume_mount_path="/runpod-volume",
        container_disk_in_gb=20,
        env={"EMBEDDER_TOKEN": "t"},
    )


def test_build_spec_forwards_template_id_and_gpu_candidates():
    from modules.embeddings.fleet import build_spec

    class _Settings:
        class runpod:
            image = "ghcr.io/x/embedder:latest"
            data_center_id = "dc"
            network_volume_id = "v"
            volume_mount_path = "/runpod-volume"
            container_disk_in_gb = 20
            pod_model_path = "/runpod-volume/models/m"
            pod_video_root = "/runpod-volume/videos"
            template_id = "tmpl_abc"
            gpu_min_ram_gb = 30

    class _Secrets:
        coordinator_public_host = "https://x.example.com"
        embedder_token = "t"
        huggingface_token = "hf"

    spec = build_spec(_Settings, _Secrets, gpu_type_ids=("L4", "4500"))
    assert spec.template_id == "tmpl_abc"
    assert spec.gpu_type_ids == ("L4", "4500")
    assert spec.min_ram_gb == 30  # RAM floor forwarded so pods honor it
    assert spec.env["ORCHESTRATOR_HOST"] == "https://x.example.com"


class _RunpodCfg:
    gpu_type_id = ""
    data_center_id = "EU-RO-1"
    gpu_max_price_hr = 0.8
    gpu_min_vram_gb = 24
    gpu_min_ram_gb = 30


def test_resolve_gpu_candidates_honors_pinned_type():
    from modules.embeddings.fleet import resolve_gpu_candidates

    class S:
        class runpod(_RunpodCfg):
            gpu_type_id = "NVIDIA L4"

    class _Client:
        def available_gpus(self, **kw):
            raise AssertionError("must not query the API when a type is pinned")

    assert resolve_gpu_candidates(S, _Client()) == ("NVIDIA L4",)


def test_resolve_gpu_candidates_auto_filters_by_price_cheapest_first():
    from core.runpod import GpuOffer
    from modules.embeddings.fleet import resolve_gpu_candidates

    class S:
        class runpod(_RunpodCfg):
            pass

    # available_gpus returns cheapest-first; B200 is over the 0.8 cap.
    offers = [
        GpuOffer("L4", "NVIDIA L4", 24, 62, 6, "Low", 0.39, None),
        GpuOffer("4000", "RTX PRO 4000", 24, 31, 12, "Low", 0.57, None),
        GpuOffer("4500", "RTX PRO 4500", 32, 62, 28, "Low", 0.74, None),
        GpuOffer("B200", "NVIDIA B200", 180, 283, 28, "Low", 5.89, None),
    ]
    seen = {}

    class _Client:
        def available_gpus(self, *, data_center_id, min_vram_gb, min_ram_gb):
            seen.update(dc=data_center_id, vram=min_vram_gb, ram=min_ram_gb)
            return offers

    cands = resolve_gpu_candidates(S, _Client())
    assert cands == ("L4", "4000", "4500")
    assert seen == {"dc": "EU-RO-1", "vram": 24, "ram": 30}


def test_reconcile_file_roundtrip(tmp_path):
    p = tmp_path / "fleet.json"
    assert read_reconcile(str(p)) == []
    write_reconcile(str(p), ["a", "b"])
    assert read_reconcile(str(p)) == ["a", "b"]


def test_fleet_deploys_persists_and_tears_down(tmp_path):
    be = _FakeBackend()
    client = RunPodClient(api_key="K", backend=be)
    path = str(tmp_path / "fleet.json")
    fleet = PodFleet(client=client, spec=_spec_kwargs(), count=2, reconcile_path=path)
    with fleet:
        fleet.ensure_started()  # deploy is deferred until remote work is known
        assert read_reconcile(path) == ["pod1", "pod2"]
        assert len(be.created) == 2
    assert set(be.terminated) == {"pod1", "pod2"}
    assert read_reconcile(path) == []  # cleared
    fleet._teardown()  # second teardown after __exit__ must be idempotent
    assert sorted(be.terminated) == ["pod1", "pod2"]  # not doubled


def test_enter_alone_deploys_nothing(tmp_path):
    """Entering the fleet must NOT deploy: a sealed or all-local rerun would
    otherwise pay for pods that lease nothing during the teardown grace."""
    be = _FakeBackend()
    client = RunPodClient(api_key="K", backend=be)
    path = str(tmp_path / "fleet.json")
    fleet = PodFleet(client=client, spec=_spec_kwargs(), count=2, reconcile_path=path)
    with fleet:
        assert be.created == []  # nothing deployed without ensure_started()
        assert read_reconcile(path) == []
    assert be.terminated == []


def test_topup_once_deploys_shortfall_and_persists(tmp_path):
    be = _FakeBackend()
    client = RunPodClient(api_key="K", backend=be)
    path = str(tmp_path / "f.json")
    fleet = PodFleet(
        client=client,
        spec=_spec_kwargs(),
        count=2,
        reconcile_path=path,
        refill=lambda: ("g",),
    )
    fleet._topup_once()
    assert fleet._ids == ["pod1", "pod2"]
    assert sorted(read_reconcile(path)) == ["pod1", "pod2"]
    fleet._topup_once()  # already at target -> no new pods
    assert len(be.created) == 2


def test_topup_once_skips_when_no_candidates(tmp_path):
    be = _FakeBackend()
    client = RunPodClient(api_key="K", backend=be)
    fleet = PodFleet(
        client=client,
        spec=_spec_kwargs(),
        count=2,
        reconcile_path=str(tmp_path / "f.json"),
        refill=lambda: (),  # nothing under cap right now
    )
    fleet._topup_once()
    assert be.created == [] and fleet._ids == []


def test_topup_once_refetches_after_stockout(tmp_path):
    # Models "no gpu now -> retry later -> stock appears". Each _topup_once is a
    # fresh re-fetch, so an empty round leaves the fleet to try again.
    class _Backend(_FakeBackend):
        in_stock = False

        def create_pod(self, **kwargs):
            if not self.in_stock:
                raise RuntimeError("no instances available")
            return super().create_pod(**kwargs)

    be = _Backend()
    client = RunPodClient(api_key="K", backend=be)
    fleet = PodFleet(
        client=client,
        spec=_spec_kwargs(),
        count=1,
        reconcile_path=str(tmp_path / "f.json"),
        refill=lambda: ("g",),
    )
    fleet._topup_once()
    assert fleet._ids == []  # out of stock -> nothing yet
    be.in_stock = True
    fleet._topup_once()
    assert fleet._ids == ["pod1"]


def test_failed_orphan_does_not_reduce_topup_shortfall(tmp_path):
    """An orphan we could not confirm terminated stays tracked for teardown, but
    it must NOT count toward the target — else the current run is left short a pod
    while a stale orphan points at a dead coordinator host."""

    class _FlakyBackend(_FakeBackend):
        def terminate_pod(self, pod_id):
            if pod_id == "orphan1":
                raise RuntimeError("transient API error")
            super().terminate_pod(pod_id)

    be = _FlakyBackend()
    client = RunPodClient(api_key="K", backend=be)
    path = str(tmp_path / "f.json")
    write_reconcile(path, ["orphan1"])  # leftover the sweep cannot confirm dead
    fleet = PodFleet(
        client=client,
        spec=_spec_kwargs(),
        count=1,
        reconcile_path=path,
        refill=lambda: ("g",),
    )
    with fleet:
        fleet._topup_once()
        # a fresh pod is deployed despite the surviving orphan
        assert fleet._ids == ["pod1"]
        assert set(read_reconcile(path)) == {"orphan1", "pod1"}


def test_background_fleet_tops_up_then_tears_down(tmp_path):
    be = _FakeBackend()
    client = RunPodClient(api_key="K", backend=be)
    path = str(tmp_path / "f.json")
    fleet = PodFleet(
        client=client,
        spec=_spec_kwargs(),
        count=2,
        reconcile_path=path,
        refill=lambda: ("g",),
        poll_s=0.01,
    )
    with fleet:
        fleet.ensure_started()
        # Poll the reconcile file, not fleet._ids: _topup_once extends _ids and
        # THEN writes the file under the lock, so a reader watching _ids can win
        # the gap between the two and read an as-yet-unwritten file (CI-only race).
        deadline = time.monotonic() + 2.0
        while len(read_reconcile(path)) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert sorted(read_reconcile(path)) == ["pod1", "pod2"]
    # teardown joined the thread and terminated every pod
    assert set(be.terminated) == {"pod1", "pod2"}
    assert read_reconcile(path) == []


def test_stop_scaling_halts_topup_without_reaping(tmp_path):
    """stop_scaling joins the background top-up thread so no new pods deploy
    during the stage's drain grace, but it must NOT terminate pods already up —
    those drain on the coordinator's 410 and teardown reaps any stragglers."""
    be = _FakeBackend()
    client = RunPodClient(api_key="K", backend=be)
    path = str(tmp_path / "f.json")
    fleet = PodFleet(
        client=client,
        spec=_spec_kwargs(),
        count=2,
        reconcile_path=path,
        refill=lambda: ("g",),
        poll_s=0.01,
    )
    with fleet:
        fleet.ensure_started()
        deadline = time.monotonic() + 2.0
        while len(fleet._ids) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        fleet.stop_scaling()
        assert fleet._thread is not None and not fleet._thread.is_alive()
        assert be.terminated == []  # stop_scaling does not reap
        assert sorted(fleet._ids) == ["pod1", "pod2"]
    # teardown on __exit__ still reaps everything
    assert set(be.terminated) == {"pod1", "pod2"}
    assert read_reconcile(path) == []


def test_stop_scaling_is_noop_before_start(tmp_path):
    """A fleet whose top-up never started (sealed/all-local rerun) has no thread
    to join; stop_scaling must be a harmless no-op rather than raise."""
    be = _FakeBackend()
    client = RunPodClient(api_key="K", backend=be)
    fleet = PodFleet(
        client=client,
        spec=_spec_kwargs(),
        count=1,
        reconcile_path=str(tmp_path / "f.json"),
        refill=lambda: ("g",),
    )
    fleet.stop_scaling()
    assert be.created == [] and be.terminated == []


def test_teardown_keeps_pods_that_failed_to_terminate(tmp_path):
    """A transient terminate failure must NOT clear the pod from reconcile —
    otherwise the still-running, billable pod is lost to the next sweep."""

    class _FlakyBackend(_FakeBackend):
        def terminate_pod(self, pod_id):
            if pod_id == "pod1":
                raise RuntimeError("transient API error")
            super().terminate_pod(pod_id)

    be = _FlakyBackend()
    client = RunPodClient(api_key="K", backend=be)
    path = str(tmp_path / "fleet.json")
    fleet = PodFleet(client=client, spec=_spec_kwargs(), count=2, reconcile_path=path)
    with fleet:
        fleet.ensure_started()
    # pod1 could not be confirmed terminated -> it stays in reconcile for the
    # next run; pod2 was confirmed -> it is gone.
    assert read_reconcile(path) == ["pod1"]


def test_orphan_sweep_keeps_pods_that_failed_to_terminate(tmp_path):
    """A failed orphan termination on enter must remain tracked so it is both
    persisted for the next run and retried on teardown — never dropped."""

    class _FlakyBackend(_FakeBackend):
        def terminate_pod(self, pod_id):
            if pod_id == "orphan1":
                raise RuntimeError("transient API error")
            super().terminate_pod(pod_id)

    be = _FlakyBackend()
    client = RunPodClient(api_key="K", backend=be)
    path = str(tmp_path / "fleet.json")
    write_reconcile(path, ["orphan1", "orphan2"])
    fleet = PodFleet(client=client, spec=_spec_kwargs(), count=1, reconcile_path=path)
    with fleet:
        fleet.ensure_started()
        # orphan1 failed, orphan2 confirmed gone, pod1 newly deployed: the
        # persisted set must carry the unconfirmed orphan plus the live pod.
        assert set(read_reconcile(path)) == {"orphan1", "pod1"}


def test_run_clip_forwards_fleet_secrets(monkeypatch):
    """run_clip must thread the RunPod fleet secrets through to the fleet hook;
    dropping them silently disables auto-deploy on every pipeline run."""
    import modules.embeddings as embeddings
    from modules.embeddings.cases import EmbeddingSecrets

    captured: dict = {}

    def _capture(settings, secrets, cases=None):
        captured["secrets"] = secrets

    monkeypatch.setattr(embeddings, "embed_clip_embeddings", _capture)

    class _Secrets:
        gemini_api_key = "g"
        embedder_token = "t"
        runpod_api_key = "rk"
        coordinator_public_host = "1.2.3.4:8765"
        huggingface_token = "hf"

    class _Settings:
        class storage:
            bucket = "b"

    embeddings.run_clip(_Settings, _Secrets)
    sec = captured["secrets"]
    assert isinstance(sec, EmbeddingSecrets)
    assert sec.runpod_api_key == "rk"
    assert sec.coordinator_public_host == "1.2.3.4:8765"
    # The forwarded bag must satisfy fleet_enabled end-to-end.
    assert fleet_enabled(_Settings, sec, count=2) is True


def test_fleet_reconciles_orphans_on_enter(tmp_path):
    be = _FakeBackend()
    client = RunPodClient(api_key="K", backend=be)
    path = str(tmp_path / "fleet.json")
    write_reconcile(path, ["orphan1", "orphan2"])  # leftover from a crashed run
    fleet = PodFleet(client=client, spec=_spec_kwargs(), count=1, reconcile_path=path)
    with fleet:
        pass
    assert "orphan1" in be.terminated and "orphan2" in be.terminated


def test_fleet_tears_down_on_exception(tmp_path):
    be = _FakeBackend()
    client = RunPodClient(api_key="K", backend=be)
    path = str(tmp_path / "fleet.json")
    fleet = PodFleet(client=client, spec=_spec_kwargs(), count=1, reconcile_path=path)
    try:
        with fleet:
            fleet.ensure_started()
            raise RuntimeError("stage blew up")
    except RuntimeError:
        pass
    assert be.terminated == ["pod1"]


def test_ensure_started_is_idempotent(tmp_path):
    be = _FakeBackend()
    client = RunPodClient(api_key="K", backend=be)
    path = str(tmp_path / "f.json")
    fleet = PodFleet(client=client, spec=_spec_kwargs(), count=1, reconcile_path=path)
    with fleet:
        fleet.ensure_started()
        fleet.ensure_started()  # second call must not deploy again
        assert len(be.created) == 1


def test_teardown_reaps_inflight_deploy(tmp_path):
    """A deploy still in flight when teardown starts must be reaped, not leaked:
    teardown waits out the in-flight create_pod and then terminates the late pod."""
    import threading

    in_deploy = threading.Event()
    release = threading.Event()

    class _SlowBackend(_FakeBackend):
        def create_pod(self, **kwargs):
            in_deploy.set()
            release.wait(3.0)
            return super().create_pod(**kwargs)

    be = _SlowBackend()
    client = RunPodClient(api_key="K", backend=be)
    path = str(tmp_path / "f.json")
    fleet = PodFleet(
        client=client,
        spec=_spec_kwargs(),
        count=1,
        reconcile_path=path,
        refill=lambda: ("g",),
    )
    deploy_thread = threading.Thread(target=fleet._topup_once)
    deploy_thread.start()
    assert in_deploy.wait(2.0)  # a create_pod is in progress

    teardown_thread = threading.Thread(target=fleet._teardown)
    teardown_thread.start()
    time.sleep(0.1)  # let teardown set _torn_down while the deploy is still running
    release.set()  # the create_pod now returns AFTER teardown began

    deploy_thread.join(3.0)
    teardown_thread.join(3.0)
    assert be.created, "the in-flight deploy did create a pod"
    assert set(be.terminated) == {"pod1"}  # and teardown reaped it
    assert fleet._ids == []
    assert read_reconcile(path) == []  # nothing left billing


def test_fleet_enabled_gating():
    class S:
        class storage:
            bucket = "b"

    class Sec:
        runpod_api_key = "k"
        embedder_token = "t"
        coordinator_public_host = "h"

    assert fleet_enabled(S, Sec, count=2) is True
    assert fleet_enabled(S, Sec, count=0) is False  # no pods
    Sec.runpod_api_key = ""
    assert fleet_enabled(S, Sec, count=2) is False  # no api key


def test_embed_stage_no_op_fleet_when_disabled(monkeypatch):
    """With the fleet disabled, embed_clip_embeddings must still open the
    StageEmbedder exactly as before (no RunPod calls)."""
    import modules.embeddings.runner as runner

    opened = {"stage": False}

    class _FakeStage:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            opened["stage"] = True
            return self

        def __exit__(self, *e):
            return False

    # StageEmbedder is imported lazily inside embed_clip_embeddings, so patch the
    # symbol on its source module (distributed), not on runner.
    monkeypatch.setattr("modules.embeddings.distributed.StageEmbedder", _FakeStage)
    monkeypatch.setattr(runner, "_run_case", lambda *a, **k: None)
    monkeypatch.setenv("RUNPOD_POD_COUNT", "0")

    class _Settings:
        class embeddings:
            gemini_enabled = False

        class storage:
            bucket = ""

        class runpod:
            reconcile_path = "/tmp/none.json"

    from modules.embeddings.cases import EmbeddingSecrets

    runner.embed_clip_embeddings(_Settings, EmbeddingSecrets(), cases=["video"])
    assert opened["stage"] is True
