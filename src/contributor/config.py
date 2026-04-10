"""Contributor config loader — reads contributor.yaml and applies env var overrides."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class RecordingConfig:
    duration_seconds: int = 15
    fps: int = 15
    resolution: tuple = (640, 360)
    hand_mask_padding_px: int = 50
    codec: str = "mp4v"


@dataclass
class HuggingFaceConfig:
    repo_id: str = ""
    token_env: str = "HF_TOKEN"


@dataclass
class S3Config:
    bucket: str = ""
    prefix: str = "hand-shuffle-contributions/"
    region: str = "us-east-1"
    access_key_env: str = "AWS_ACCESS_KEY_ID"
    secret_key_env: str = "AWS_SECRET_ACCESS_KEY"


@dataclass
class StorageConfig:
    backend: str = "local"
    local_dir: str = "user-data"
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    s3: S3Config = field(default_factory=S3Config)


@dataclass
class VideoIdsConfig:
    start_index: int = 20


@dataclass
class ContributorConfig:
    collection_mode: str = "both"
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    video_ids: VideoIdsConfig = field(default_factory=VideoIdsConfig)


def load_config(yaml_path=None):
    if yaml_path is None:
        yaml_path = PROJECT_ROOT / "configs" / "contributor.yaml"

    cfg = ContributorConfig()

    if Path(yaml_path).exists():
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
        c = data.get("contributor", {})

        cfg.collection_mode = c.get("collection_mode", cfg.collection_mode)

        rec = c.get("recording", {})
        cfg.recording.duration_seconds = rec.get("duration_seconds", cfg.recording.duration_seconds)
        cfg.recording.fps = rec.get("fps", cfg.recording.fps)
        res = rec.get("resolution")
        if res:
            cfg.recording.resolution = tuple(res)
        cfg.recording.hand_mask_padding_px = rec.get("hand_mask_padding_px", cfg.recording.hand_mask_padding_px)
        cfg.recording.codec = rec.get("codec", cfg.recording.codec)

        stor = c.get("storage", {})
        cfg.storage.backend = stor.get("backend", cfg.storage.backend)
        cfg.storage.local_dir = stor.get("local_dir", cfg.storage.local_dir)

        hf = stor.get("huggingface", {})
        cfg.storage.huggingface.repo_id = hf.get("repo_id", cfg.storage.huggingface.repo_id)
        cfg.storage.huggingface.token_env = hf.get("token_env", cfg.storage.huggingface.token_env)

        s3 = stor.get("s3", {})
        cfg.storage.s3.bucket = s3.get("bucket", cfg.storage.s3.bucket)
        cfg.storage.s3.prefix = s3.get("prefix", cfg.storage.s3.prefix)
        cfg.storage.s3.region = s3.get("region", cfg.storage.s3.region)
        cfg.storage.s3.access_key_env = s3.get("access_key_env", cfg.storage.s3.access_key_env)
        cfg.storage.s3.secret_key_env = s3.get("secret_key_env", cfg.storage.s3.secret_key_env)

        vid = c.get("video_ids", {})
        cfg.video_ids.start_index = vid.get("start_index", cfg.video_ids.start_index)

    if os.environ.get("CONTRIBUTOR_COLLECTION_MODE"):
        cfg.collection_mode = os.environ["CONTRIBUTOR_COLLECTION_MODE"]
    if os.environ.get("CONTRIBUTOR_STORAGE_BACKEND"):
        cfg.storage.backend = os.environ["CONTRIBUTOR_STORAGE_BACKEND"]
    if os.environ.get("CONTRIBUTOR_LOCAL_DIR"):
        cfg.storage.local_dir = os.environ["CONTRIBUTOR_LOCAL_DIR"]
    if os.environ.get("CONTRIBUTOR_HF_REPO_ID"):
        cfg.storage.huggingface.repo_id = os.environ["CONTRIBUTOR_HF_REPO_ID"]

    return cfg
