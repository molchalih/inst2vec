#!/usr/bin/env python
"""List RunPod GPU types that can attach to your network volume right now.

Pins the query to the volume's datacenter (volumes are region-locked, so a pod
must launch in the same DC to mount it), then lists in-stock GPUs meeting the
VRAM/RAM floors and marks the ones within the price cap that the fleet would
auto-deploy. Floors/cap default to the live config (config.toml + .env).

    uv run python scripts/runpod_gpus.py
    uv run python scripts/runpod_gpus.py --min-vram 32 --max-price 1.0
    uv run python scripts/runpod_gpus.py --data-center EU-RO-1   # skip lookup
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv

load_dotenv()

from core.config import load_runpod_config  # noqa: E402
from core.runpod import RunPodClient  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-vram", type=int, help="min GPU VRAM (GB)")
    ap.add_argument("--min-ram", type=int, help="min system RAM (GB)")
    ap.add_argument("--max-price", type=float, help="price cap $/hr (fleet filter)")
    ap.add_argument(
        "--data-center",
        default="",
        help="datacenter id; defaults to the network volume's DC",
    )
    args = ap.parse_args()

    settings, runpod_api_key = load_runpod_config()
    if not runpod_api_key:
        print("RUNPOD_API_KEY is not set in your environment / .env", file=sys.stderr)
        return 1

    # Default the floors/cap to what the fleet actually uses, so this view
    # matches which GPUs auto-fetch would deploy.
    rp = settings.runpod
    min_vram = args.min_vram if args.min_vram is not None else rp.gpu_min_vram_gb
    min_ram = args.min_ram if args.min_ram is not None else rp.gpu_min_ram_gb
    max_price = args.max_price if args.max_price is not None else rp.gpu_max_price_hr

    client = RunPodClient(api_key=runpod_api_key)

    data_center = args.data_center or settings.runpod.data_center_id
    if not data_center:
        volume_id = settings.runpod.network_volume_id
        if not volume_id:
            print(
                "No datacenter known: set RUNPOD_NETWORK_VOLUME_ID in .env (to "
                "resolve it from the volume) or pass --data-center.",
                file=sys.stderr,
            )
            return 1
        data_center = client.network_volume_datacenter(volume_id)
        if not data_center:
            print(
                f"Network volume {volume_id!r} not found on this account.",
                file=sys.stderr,
            )
            return 1
        print(f"Volume {volume_id} lives in datacenter {data_center}\n")

    offers = client.available_gpus(
        data_center_id=data_center,
        min_vram_gb=min_vram,
        min_ram_gb=min_ram,
    )
    if not offers:
        print(
            f"No GPUs in {data_center} are in stock with >={min_vram}GB VRAM / "
            f">={min_ram}GB RAM right now. Lower the floors or retry later."
        )
        return 0

    header = f"{'':<2}{'GPU type id':<28} {'VRAM':>5} {'RAM':>5} {'vCPU':>5} {'stock':>7} {'$/hr':>7}  free"
    print(header)
    print("-" * len(header))
    for o in offers:
        price = f"{o.price_hr:.3f}" if o.price_hr is not None else "?"
        ram = f"{o.ram_gb}" if o.ram_gb is not None else "?"
        vcpu = f"{o.vcpu}" if o.vcpu is not None else "?"
        free = o.max_unreserved if o.max_unreserved is not None else "?"
        # Mark rows the fleet would actually try (within the price cap).
        within = o.price_hr is not None and o.price_hr <= max_price
        print(
            f"{'* ' if within else '  '}{o.id:<28} {o.vram_gb:>4}G {ram:>4}G "
            f"{vcpu:>5} {o.stock_status or '?':>7} {price:>7}  {free}"
        )

    candidates = [
        o for o in offers if o.price_hr is not None and o.price_hr <= max_price
    ]
    print(f"\n* = within the ${max_price:.2f}/hr cap — what the fleet auto-deploys.")
    if candidates:
        ids = ", ".join(o.id for o in candidates)
        print(f"Fleet will try (cheapest-first): {ids}")
        print('Pin one with RUNPOD_GPU_TYPE_ID="…" in .env to override (optional).')
    else:
        print(
            f"Nothing in stock under ${max_price:.2f}/hr right now → the fleet would "
            "deploy 0 pods and embed on the orchestrator GPU only.\n"
            "Raise RUNPOD_GPU_MAX_PRICE_HR in .env or retry later."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
