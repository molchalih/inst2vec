"""GPU pod inference service.

Runs the local Qwen3-VL embedder as a pull-worker that leases
clip-embedding jobs from the orchestrator coordinator and embeds them
on a rented GPU pod.
"""
