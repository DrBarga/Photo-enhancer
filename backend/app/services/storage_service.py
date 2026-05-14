from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests


@dataclass
class StoredObject:
    key: str
    uri: str
    public_url: str | None = None


class StorageService:
    def put_png(self, key: str, image_rgb: np.ndarray) -> StoredObject:
        return self.put_bytes(key, _png_bytes(image_rgb), "image/png")

    def put_bytes(self, key: str, content: bytes, content_type: str) -> StoredObject:
        raise NotImplementedError

    def read_bytes(self, key: str) -> bytes:
        raise NotImplementedError

    def read_image(self, key: str) -> np.ndarray:
        payload = self.read_bytes(key)
        data = np.frombuffer(payload, dtype=np.uint8)
        image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(key)
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


class LocalStorageService(StorageService):
    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, key: str, content: bytes, content_type: str) -> StoredObject:
        path = self.base_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return StoredObject(key=key, uri=str(path), public_url=None)

    def read_bytes(self, key: str) -> bytes:
        path = Path(key)
        if not path.is_absolute():
            path = self.base_dir / key
        return path.read_bytes()


class SupabaseStorageService(StorageService):
    def __init__(self) -> None:
        self.url = os.getenv("SUPABASE_URL", "").rstrip("/")
        self.key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        self.bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "photo-enhancer")
        if not self.url or not self.key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")

    def put_bytes(self, key: str, content: bytes, content_type: str) -> StoredObject:
        endpoint = f"{self.url}/storage/v1/object/{self.bucket}/{key}"
        response = requests.post(
            endpoint,
            headers={
                "authorization": f"Bearer {self.key}",
                "apikey": self.key,
                "content-type": content_type,
                "x-upsert": "true",
            },
            data=content,
            timeout=60,
        )
        response.raise_for_status()
        public_url = f"{self.url}/storage/v1/object/public/{self.bucket}/{key}"
        return StoredObject(key=key, uri=f"supabase://{self.bucket}/{key}", public_url=public_url)

    def read_bytes(self, key: str) -> bytes:
        key = key.replace(f"supabase://{self.bucket}/", "")
        endpoint = f"{self.url}/storage/v1/object/{self.bucket}/{key}"
        response = requests.get(
            endpoint,
            headers={"authorization": f"Bearer {self.key}", "apikey": self.key},
            timeout=60,
        )
        response.raise_for_status()
        return response.content


class S3StorageService(StorageService):
    def __init__(self) -> None:
        import boto3

        self.bucket = os.getenv("AI_LIGHT_S3_BUCKET", "")
        self.public_base_url = os.getenv("AI_LIGHT_S3_PUBLIC_BASE_URL", "").rstrip("/")
        if not self.bucket:
            raise RuntimeError("AI_LIGHT_S3_BUCKET is required")
        self.client = boto3.client(
            "s3",
            endpoint_url=os.getenv("AI_LIGHT_S3_ENDPOINT_URL") or None,
            aws_access_key_id=os.getenv("AI_LIGHT_S3_ACCESS_KEY_ID") or None,
            aws_secret_access_key=os.getenv("AI_LIGHT_S3_SECRET_ACCESS_KEY") or None,
            region_name=os.getenv("AI_LIGHT_S3_REGION") or "auto",
        )

    def put_bytes(self, key: str, content: bytes, content_type: str) -> StoredObject:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=content, ContentType=content_type)
        public_url = f"{self.public_base_url}/{key}" if self.public_base_url else None
        return StoredObject(key=key, uri=f"s3://{self.bucket}/{key}", public_url=public_url)

    def read_bytes(self, key: str) -> bytes:
        key = key.replace(f"s3://{self.bucket}/", "")
        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read()


def create_storage_service(local_base_dir: str) -> StorageService:
    provider = os.getenv("AI_LIGHT_STORAGE_PROVIDER", "local").strip().lower()
    if provider == "supabase":
        return SupabaseStorageService()
    if provider in {"s3", "r2"}:
        return S3StorageService()
    return LocalStorageService(local_base_dir)


def _png_bytes(image_rgb: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", cv2.cvtColor(np.clip(image_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("Could not encode PNG")
    return encoded.tobytes()
