from core.runpod import GpuOffer, PodSpec, RunPodClient


class _FakeBackend:
    def __init__(self):
        self.api_key = None
        self.created = []
        self.terminated = []
        self._pods = []

    def create_pod(self, **kwargs):
        self.created.append(kwargs)
        pod = {"id": f"pod{len(self.created)}", "name": kwargs["name"]}
        self._pods.append(pod)
        return pod

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)
        self._pods = [p for p in self._pods if p["id"] != pod_id]

    def get_pods(self):
        return list(self._pods)


def _spec():
    return PodSpec(
        image="reg/inst2vec-embedder:latest",
        gpu_type_ids=("NVIDIA GeForce RTX 4090",),
        data_center_id="EU-RO-1",
        network_volume_id="vol123",
        volume_mount_path="/runpod-volume",
        container_disk_in_gb=20,
        env={"EMBEDDER_TOKEN": "t", "ORCHESTRATOR_HOST": "1.2.3.4:8765"},
        name_prefix="inst2vec-embed",
    )


def test_deploy_sets_api_key_and_creates_n_pods():
    be = _FakeBackend()
    client = RunPodClient(api_key="KEY", backend=be)
    ids = client.deploy(3, _spec())
    assert be.api_key == "KEY"
    assert ids == ["pod1", "pod2", "pod3"]
    assert len(be.created) == 3
    first = be.created[0]
    assert first["image_name"] == "reg/inst2vec-embedder:latest"
    assert first["gpu_type_id"] == "NVIDIA GeForce RTX 4090"
    assert first["network_volume_id"] == "vol123"
    assert first["volume_mount_path"] == "/runpod-volume"
    assert first["data_center_id"] == "EU-RO-1"
    assert first["env"]["EMBEDDER_TOKEN"] == "t"
    assert first["name"].startswith("inst2vec-embed")


def test_deploy_passes_ram_floor_as_min_memory():
    # The candidate scan filters offers by system RAM, but RunPod can still place
    # the chosen GPU type on a lower-RAM instance unless create_pod is told the
    # floor — pass it through so the deployed pod honors it.
    be = _FakeBackend()
    client = RunPodClient(api_key="KEY", backend=be)
    spec = _spec()
    spec.min_ram_gb = 30
    client.deploy(1, spec)
    assert be.created[0]["min_memory_in_gb"] == 30


def test_deploy_falls_back_to_next_gpu_when_first_unavailable():
    # Low-stock GPUs: the first candidate is out, so deploy must try the next
    # rather than fail the whole pod.
    class _PickyBackend(_FakeBackend):
        def create_pod(self, **kwargs):
            if kwargs["gpu_type_id"] == "OUT_OF_STOCK":
                raise RuntimeError("no instances available")
            return super().create_pod(**kwargs)

    be = _PickyBackend()
    client = RunPodClient(api_key="KEY", backend=be)
    spec = _spec()
    spec.gpu_type_ids = ("OUT_OF_STOCK", "NVIDIA L4")
    ids = client.deploy(1, spec)
    assert ids == ["pod1"]
    assert be.created[0]["gpu_type_id"] == "NVIDIA L4"


def test_deploy_returns_empty_when_all_gpu_candidates_unavailable():
    class _OosBackend(_FakeBackend):
        def create_pod(self, **kwargs):
            raise RuntimeError("no instances available")

    client = RunPodClient(api_key="KEY", backend=_OosBackend())
    spec = _spec()
    spec.gpu_type_ids = ("a", "b")
    assert client.deploy(2, spec) == []  # degrade to local-only, never raise


def test_deploy_with_template_id_uses_template_not_image():
    # Private image: the registry credential lives on a RunPod template, so we
    # launch from the template id and must NOT pass image_name (which would have
    # no pull auth).
    be = _FakeBackend()
    client = RunPodClient(api_key="KEY", backend=be)
    spec = _spec()
    spec.template_id = "tmpl_abc"
    client.deploy(1, spec)
    created = be.created[0]
    assert created["template_id"] == "tmpl_abc"
    assert "image_name" not in created
    # pod-level placement + dynamic env still come from the spec, not the template
    assert created["gpu_type_id"] == "NVIDIA GeForce RTX 4090"
    assert created["network_volume_id"] == "vol123"
    assert created["env"]["EMBEDDER_TOKEN"] == "t"


def test_terminate_is_idempotent_and_swallows_unknown():
    be = _FakeBackend()
    client = RunPodClient(api_key="KEY", backend=be)
    ids = client.deploy(1, _spec())
    client.terminate([*ids, "ghost"])  # ghost already gone -> must not raise
    assert "pod1" in be.terminated


def test_terminate_continues_after_one_failure_and_reports_it():
    class _FlakyBackend(_FakeBackend):
        def terminate_pod(self, pod_id):
            if pod_id == "pod1":
                raise RuntimeError("transient")
            super().terminate_pod(pod_id)

    be = _FlakyBackend()
    client = RunPodClient(api_key="KEY", backend=be)
    client.deploy(2, _spec())
    failed = client.terminate(
        ["pod1", "pod2"]
    )  # must attempt pod2 despite pod1 raising
    assert "pod2" in be.terminated
    # The unconfirmed pod is reported back so the caller can keep it for the
    # next reconcile sweep instead of leaking a billable pod.
    assert failed == ["pod1"]


def test_terminate_returns_empty_when_all_confirmed():
    be = _FakeBackend()
    client = RunPodClient(api_key="KEY", backend=be)
    client.deploy(2, _spec())
    assert client.terminate(["pod1", "pod2"]) == []


def test_list_ids():
    be = _FakeBackend()
    client = RunPodClient(api_key="KEY", backend=be)
    client.deploy(2, _spec())
    assert set(client.list_ids()) == {"pod1", "pod2"}


def _gpu_response():
    def lp(stock, ram, vcpu, price, unreserved):
        return {
            "stockStatus": stock,
            "minMemory": ram,
            "minVcpu": vcpu,
            "uninterruptablePrice": price,
            "maxUnreservedGpuCount": unreserved,
        }

    return {
        "data": {
            "gpuTypes": [
                {  # in stock, fits -> kept
                    "id": "A",
                    "displayName": "RTX 4090",
                    "memoryInGb": 24,
                    "lowestPrice": lp("High", 41, 8, 0.69, 3),
                },
                {  # no stock in this DC -> dropped
                    "id": "B",
                    "displayName": "RTX 3090",
                    "memoryInGb": 24,
                    "lowestPrice": lp(None, None, None, None, None),
                },
                {  # in stock, fits, cheaper -> kept and sorts first
                    "id": "C",
                    "displayName": "A40",
                    "memoryInGb": 48,
                    "lowestPrice": lp("Low", 50, 9, 0.39, 1),
                },
                {  # VRAM below floor -> dropped
                    "id": "D",
                    "displayName": "Tiny",
                    "memoryInGb": 16,
                    "lowestPrice": lp("High", 32, 4, 0.2, 5),
                },
            ]
        }
    }


def test_available_gpus_filters_by_stock_and_vram_sorted_by_price():
    captured = {}

    def fake_query(query, api_key=None):
        captured["query"] = query
        captured["api_key"] = api_key
        return _gpu_response()

    client = RunPodClient(api_key="KEY", query_fn=fake_query)
    offers = client.available_gpus(
        data_center_id="EU-RO-1", min_vram_gb=24, min_ram_gb=30
    )

    # B dropped (no stock in DC), D dropped (VRAM < 24); cheaper offer first.
    assert [o.id for o in offers] == ["C", "A"]
    assert isinstance(offers[0], GpuOffer)
    assert offers[0].vram_gb == 48
    assert offers[0].ram_gb == 50
    assert offers[0].stock_status == "Low"
    # The query pins the volume's datacenter and forwards deploy constraints.
    assert 'dataCenterId: "EU-RO-1"' in captured["query"]
    assert "minMemoryInGb: 30" in captured["query"]
    assert "secureCloud: true" in captured["query"]
    assert "supportPublicIp: true" in captured["query"]
    assert captured["api_key"] == "KEY"


def test_network_volume_datacenter_resolves_from_id():
    def fake_query(query, api_key=None):
        return {
            "data": {
                "myself": {
                    "networkVolumes": [
                        {"id": "volX", "dataCenterId": "EU-RO-1"},
                        {"id": "volY", "dataCenterId": "US-OR-1"},
                    ]
                }
            }
        }

    client = RunPodClient(api_key="KEY", query_fn=fake_query)
    assert client.network_volume_datacenter("volY") == "US-OR-1"
    assert client.network_volume_datacenter("ghost") is None
