"""GPU pod inference service.

Wraps the local Qwen3-VL embedder behind a FastAPI HTTP API so the
clip-embeddings pipeline stage can run on a rented GPU pod.
"""
