"""Thin RunPod API client for the embedding pod fleet.

Wraps the ``runpod`` SDK behind an injectable ``backend`` so the rest of the
codebase (and tests) never import the SDK directly. Knows nothing about
embeddings — it only deploys, terminates, and lists pods.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from core.console import log


@dataclass
class PodSpec:
    image: str
    # Ordered GPU types to try per pod, cheapest-first. deploy() falls through
    # to the next when a type is out of stock, so a Low-stock DC still deploys.
    gpu_type_ids: tuple[str, ...]
    data_center_id: str
    network_volume_id: str
    volume_mount_path: str
    container_disk_in_gb: int
    env: dict[str, str]
    # System-RAM floor passed to create_pod as min_memory_in_gb, so RunPod
    # cannot place the chosen GPU type on an instance below it (mixed-RAM types
    # otherwise OOM loading the model). 0 -> no floor (the SDK default).
    min_ram_gb: int = 0
    name_prefix: str = "inst2vec-embed"
    # When set, deploy from this RunPod template (carries the private-registry
    # pull credential) instead of ``image`` — the SDK cannot pass registry auth
    # on a bare image. Empty -> deploy from ``image`` (public registry).
    template_id: str = ""


@dataclass
class GpuOffer:
    """A GPU type with its current on-demand availability in one datacenter."""

    id: str
    display_name: str
    vram_gb: int
    ram_gb: int | None
    vcpu: int | None
    stock_status: str | None
    price_hr: float | None
    max_unreserved: int | None


def _default_backend():
    import runpod  # imported lazily so the SDK is only needed when deploying

    return runpod


_VOLUME_QUERY = "query myself { myself { networkVolumes { id dataCenterId } } }"


def _gpu_availability_query(
    *,
    data_center_id: str | None,
    gpu_count: int,
    min_ram_gb: int,
    secure: bool,
    support_public_ip: bool,
) -> str:
    """Build a ``gpuTypes`` query whose ``lowestPrice`` input mirrors the
    constraints the fleet deploys with, so an offer that comes back is one the
    pod can actually claim against the (region-pinned) network volume."""
    inp = [f"gpuCount: {gpu_count}"]
    if data_center_id:
        inp.append(f'dataCenterId: "{data_center_id}"')
    if min_ram_gb:
        inp.append(f"minMemoryInGb: {min_ram_gb}")
    inp.append(f"secureCloud: {str(secure).lower()}")
    inp.append(f"supportPublicIp: {str(support_public_ip).lower()}")
    return f"""
    query GpuAvailability {{
      gpuTypes {{
        id
        displayName
        memoryInGb
        lowestPrice(input: {{{", ".join(inp)}}}) {{
          stockStatus
          minMemory
          minVcpu
          uninterruptablePrice
          maxUnreservedGpuCount
        }}
      }}
    }}
    """


class RunPodClient:
    def __init__(self, *, api_key: str, backend=None, query_fn=None) -> None:
        self._backend = backend if backend is not None else _default_backend()
        self._backend.api_key = api_key
        self._api_key = api_key
        self._query_fn = query_fn

    def _query(self, query: str) -> dict:
        fn = self._query_fn
        if fn is None:  # lazily bind the SDK's GraphQL runner outside tests
            from runpod.api.graphql import run_graphql_query

            fn = run_graphql_query
        return fn(query, api_key=self._api_key)

    def network_volume_datacenter(self, volume_id: str) -> str | None:
        """Return the datacenter id a network volume lives in, or None if the
        volume is not found under this account."""
        data = self._query(_VOLUME_QUERY)
        volumes = (data.get("data", {}).get("myself", {}) or {}).get(
            "networkVolumes"
        ) or []
        for vol in volumes:
            if vol.get("id") == volume_id:
                return vol.get("dataCenterId")
        return None

    def available_gpus(
        self,
        *,
        data_center_id: str | None,
        gpu_count: int = 1,
        min_vram_gb: int = 0,
        min_ram_gb: int = 0,
        secure: bool = True,
        support_public_ip: bool = True,
    ) -> list[GpuOffer]:
        """List GPU types deployable in ``data_center_id`` right now, filtered to
        those with at least ``min_vram_gb`` VRAM / ``min_ram_gb`` system RAM.

        A GPU with no ``stockStatus`` has no capacity in that datacenter under
        the given constraints and is dropped. Results are sorted cheapest-first.
        """
        query = _gpu_availability_query(
            data_center_id=data_center_id,
            gpu_count=gpu_count,
            min_ram_gb=min_ram_gb,
            secure=secure,
            support_public_ip=support_public_ip,
        )
        data = self._query(query)
        offers: list[GpuOffer] = []
        for gpu in data.get("data", {}).get("gpuTypes") or []:
            vram = gpu.get("memoryInGb") or 0
            if vram < min_vram_gb:
                continue
            price = gpu.get("lowestPrice") or {}
            if not price.get("stockStatus"):  # null -> unavailable in this DC
                continue
            offers.append(
                GpuOffer(
                    id=gpu["id"],
                    display_name=gpu.get("displayName", gpu["id"]),
                    vram_gb=vram,
                    ram_gb=price.get("minMemory"),
                    vcpu=price.get("minVcpu"),
                    stock_status=price.get("stockStatus"),
                    price_hr=price.get("uninterruptablePrice"),
                    max_unreserved=price.get("maxUnreservedGpuCount"),
                )
            )
        offers.sort(key=lambda o: (o.price_hr is None, o.price_hr or 0.0))
        return offers

    def deploy(self, n: int, spec: PodSpec) -> list[str]:
        ids: list[str] = []
        for _ in range(n):
            pod = self._create_one(spec)
            if pod is not None:
                ids.append(pod["id"])
                log("fleet", "PUT", pod["id"], "deployed")
        return ids

    def _create_one(self, spec: PodSpec) -> dict | None:
        """Create one pod, trying each GPU type in order. Returns None (never
        raises) when every candidate is unavailable, so a stock shortage
        degrades to fewer pods rather than aborting the run."""
        name = f"{spec.name_prefix}-{uuid.uuid4().hex[:8]}"
        common = dict(
            name=name,
            cloud_type="SECURE",
            data_center_id=spec.data_center_id,
            network_volume_id=spec.network_volume_id,
            volume_mount_path=spec.volume_mount_path,
            container_disk_in_gb=spec.container_disk_in_gb,
            min_memory_in_gb=spec.min_ram_gb,
            env=dict(spec.env),
            support_public_ip=True,
        )
        # A template carries the private-registry pull credential and its own
        # image; passing image_name alongside it would have no auth.
        if spec.template_id:
            common["template_id"] = spec.template_id
        else:
            common["image_name"] = spec.image
        for gpu_type_id in spec.gpu_type_ids:
            try:
                return self._backend.create_pod(gpu_type_id=gpu_type_id, **common)
            except Exception as exc:  # out of stock / transient — try the next type
                log(
                    "fleet",
                    "PUT",
                    gpu_type_id,
                    "WARN",
                    stats={"deploy_err": repr(exc)},
                )
        if spec.gpu_type_ids:
            log("fleet", "PUT", "-", "FAIL", stats={"tried": len(spec.gpu_type_ids)})
        return None

    def terminate(self, pod_ids: Iterable[str]) -> list[str]:
        """Terminate each pod; return the IDs whose termination was NOT confirmed.

        A failure (transient API error or unknown pod) is logged but never
        raised, so teardown is never blocked. The unconfirmed IDs are returned
        so the caller can keep them in the reconcile state and retry next run
        rather than leaking a still-billing pod."""
        failed: list[str] = []
        for pod_id in pod_ids:
            try:
                self._backend.terminate_pod(pod_id)
                log("fleet", "SWEEP", pod_id, "terminated")
            except Exception as exc:  # already-gone or transient — never block teardown
                failed.append(pod_id)
                log(
                    "fleet", "SWEEP", pod_id, "WARN", stats={"terminate_err": repr(exc)}
                )
        return failed

    def list_ids(self) -> list[str]:
        return [p["id"] for p in self._backend.get_pods()]
