from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


@dataclass
class Config:
    seed: int = 42
    data: dict = None
    modeling: dict = None
    simulation: dict = None

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "Config":
        if config_path is None:
            config_path = project_root() / "config" / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        c = cls(
            seed=cfg.get("seed", 42),
            data=cfg.get("data", {}),
            modeling=cfg.get("modeling", {}),
            simulation=cfg.get("simulation", {}),
        )
        return c

    def path(self, key: str) -> Path:
        d = self.data
        if key == "raw_dir":
            return project_root() / d.get("raw_dir", "data/raw")
        if key == "processed_dir":
            return project_root() / d.get("processed_dir", "data/processed")
        if key == "accepted_csv":
            return project_root() / d.get("accepted_csv", "data/raw/accepted_2007_to_2018Q4.csv")
        if key == "rejected_csv":
            return project_root() / d.get("rejected_csv", "data/raw/rejected_2007_to_2018Q4.csv")
        if key == "accepted_parquet":
            return project_root() / d.get("accepted_parquet", "data/processed/accepted.parquet")
        if key == "modeling_parquet":
            return project_root() / d.get("modeling_parquet", "data/processed/modeling_sample.parquet")
        raise KeyError(key)
