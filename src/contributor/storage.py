"""Storage backend ABC and implementations: local, HuggingFace, S3."""

import csv
import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class StorageBackend(ABC):
    @abstractmethod
    def save_keypoints(self, video_id, kp_array, mask_array, meta): ...

    @abstractmethod
    def save_video(self, video_id, video_path): ...

    @abstractmethod
    def save_consent(self, consent_id, consent_data): ...

    @abstractmethod
    def save_labels(self, video_id, start_hand, end_hand, session_id, timestamp): ...


class LocalStorage(StorageBackend):
    def __init__(self, root_dir):
        self.root = Path(root_dir)
        self._label_lock = threading.Lock()
        for d in ("consents", "keypoints", "videos", "labels"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def save_keypoints(self, video_id, kp_array, mask_array, meta):
        kp_dir = self.root / "keypoints"
        np.save(kp_dir / f"{video_id}.npy", kp_array)
        np.save(kp_dir / f"{video_id}_mask.npy", mask_array)
        np.save(kp_dir / f"{video_id}_meta.npy", meta, allow_pickle=True)

    def save_video(self, video_id, video_path):
        import shutil
        dst = self.root / "videos" / f"{video_id}.mp4"
        shutil.move(str(video_path), str(dst))
        return dst

    def save_consent(self, consent_id, consent_data):
        ts = consent_data.get("timestamp", "")[:10]
        path = self.root / "consents" / f"{ts}_{consent_id}.json"
        with open(path, "w") as f:
            json.dump(consent_data, f, indent=2)

    def save_labels(self, video_id, start_hand, end_hand, session_id, timestamp):
        csv_path = self.root / "labels" / "contributions.csv"
        with self._label_lock:
            write_header = not csv_path.exists()
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["video_id", "start_hand", "end_hand", "session_id", "timestamp"])
                writer.writerow([video_id, start_hand, end_hand, session_id, timestamp])


class HuggingFaceStorage(StorageBackend):
    def __init__(self, cfg, local_tmp="user-data-tmp"):
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for HuggingFace storage. "
                "Install with: pip install huggingface_hub"
            )
        import os
        token = os.environ.get(cfg.token_env)
        if not token:
            raise ValueError(f"HuggingFace token not found in env var: {cfg.token_env}")
        self.api = HfApi(token=token)
        self.repo_id = cfg.repo_id
        self._local = LocalStorage(local_tmp)

    def _upload(self, local_path, remote_path):
        self.api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=self.repo_id,
            repo_type="dataset",
        )

    def save_keypoints(self, video_id, kp_array, mask_array, meta):
        self._local.save_keypoints(video_id, kp_array, mask_array, meta)
        for suffix in ("", "_mask", "_meta"):
            p = self._local.root / "keypoints" / f"{video_id}{suffix}.npy"
            self._upload(p, f"keypoints/{video_id}{suffix}.npy")

    def save_video(self, video_id, video_path):
        self._upload(video_path, f"videos/{video_id}.mp4")

    def save_consent(self, consent_id, consent_data):
        self._local.save_consent(consent_id, consent_data)
        ts = consent_data.get("timestamp", "")[:10]
        p = self._local.root / "consents" / f"{ts}_{consent_id}.json"
        self._upload(p, f"consents/{ts}_{consent_id}.json")

    def save_labels(self, video_id, start_hand, end_hand, session_id, timestamp):
        self._local.save_labels(video_id, start_hand, end_hand, session_id, timestamp)
        csv_path = self._local.root / "labels" / "contributions.csv"
        self._upload(csv_path, "labels/contributions.csv")


class S3Storage(StorageBackend):
    def __init__(self, cfg, local_tmp="user-data-tmp"):
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 storage. Install with: pip install boto3"
            )
        import os
        self.s3 = boto3.client(
            "s3",
            region_name=cfg.region,
            aws_access_key_id=os.environ.get(cfg.access_key_env),
            aws_secret_access_key=os.environ.get(cfg.secret_key_env),
        )
        self.bucket = cfg.bucket
        self.prefix = cfg.prefix
        self._local = LocalStorage(local_tmp)

    def _upload(self, local_path, remote_key):
        self.s3.upload_file(str(local_path), self.bucket, self.prefix + remote_key)

    def save_keypoints(self, video_id, kp_array, mask_array, meta):
        self._local.save_keypoints(video_id, kp_array, mask_array, meta)
        for suffix in ("", "_mask", "_meta"):
            p = self._local.root / "keypoints" / f"{video_id}{suffix}.npy"
            self._upload(p, f"keypoints/{video_id}{suffix}.npy")

    def save_video(self, video_id, video_path):
        self._upload(video_path, f"videos/{video_id}.mp4")

    def save_consent(self, consent_id, consent_data):
        self._local.save_consent(consent_id, consent_data)
        ts = consent_data.get("timestamp", "")[:10]
        p = self._local.root / "consents" / f"{ts}_{consent_id}.json"
        self._upload(p, f"consents/{ts}_{consent_id}.json")

    def save_labels(self, video_id, start_hand, end_hand, session_id, timestamp):
        self._local.save_labels(video_id, start_hand, end_hand, session_id, timestamp)
        csv_path = self._local.root / "labels" / "contributions.csv"
        self._upload(csv_path, "labels/contributions.csv")


def make_storage(cfg):
    backend = cfg.storage.backend
    if backend == "local":
        return LocalStorage(cfg.storage.local_dir)
    elif backend == "huggingface":
        return HuggingFaceStorage(cfg.storage.huggingface)
    elif backend == "s3":
        return S3Storage(cfg.storage.s3)
    else:
        raise ValueError(f"Unknown storage backend: {backend!r}")
