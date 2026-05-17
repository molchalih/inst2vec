"""S3-compatible object store wrapper for the embedder data plane.

Wraps boto3 with a small interface: ``put`` (idempotent upload),
``head`` (existence + size), ``signed_get`` (pre-signed URL the GPU pod
can fetch). Endpoint URL drives backend choice — AWS S3, Cloudflare R2,
Backblaze B2, MinIO all speak the same dialect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import boto3
from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from modules.config import StorageSettings


@dataclass
class ObjectMeta:
    size: int
    etag: str


class ObjectStore:
    """S3-compatible store. Created once per pipeline run."""

    def __init__(
        self,
        *,
        settings: StorageSettings,
        endpoint_url: str | None,
        access_key: str,
        secret_key: str,
    ) -> None:
        self.settings = settings
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url or None,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    # ── key helpers ──────────────────────────────────────────────────────────

    def key_for_clip(self, clip_id: int) -> str:
        prefix = self.settings.prefix.rstrip("/")
        return f"{prefix}/{clip_id}.mp4" if prefix else f"{clip_id}.mp4"

    # ── operations ───────────────────────────────────────────────────────────

    def head(self, key: str) -> dict | None:
        try:
            r = self.client.head_object(Bucket=self.settings.bucket, Key=key)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                return None
            raise
        return {"size": int(r["ContentLength"]), "etag": r["ETag"]}

    def put(self, local_path: str, key: str) -> None:
        self.client.upload_file(local_path, self.settings.bucket, key)

    def signed_get(self, key: str, ttl_s: int | None = None) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.settings.bucket, "Key": key},
            ExpiresIn=ttl_s or self.settings.signed_url_ttl_s,
        )


def get_object_store(settings, secrets) -> ObjectStore:
    """Construct an ObjectStore from the runtime settings + secrets."""
    return ObjectStore(
        settings=settings.storage,
        endpoint_url=secrets.object_store_endpoint or None,
        access_key=secrets.object_store_access_key,
        secret_key=secrets.object_store_secret_key,
    )
