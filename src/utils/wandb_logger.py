"""Minimal wandb wrapper: on/off/offline toggle, no-op when disabled."""
from __future__ import annotations

import os
from typing import Any, Mapping


class WandbLogger:
    """Thin wrapper. If enabled=False, every method is a no-op."""

    def __init__(
        self,
        enabled: bool,
        project: str | None,
        run_name: str | None,
        mode: str | None,
        config: Mapping[str, Any] | None,
        tags: list[str] | None = None,
        group: str | None = None,
        dir_: str | None = None,
    ):
        self.enabled = bool(enabled)
        self._run = None
        if not self.enabled:
            return

        import wandb  # lazy import

        if dir_ is not None:
            os.makedirs(dir_, exist_ok=True)

        kwargs: dict[str, Any] = {"project": project, "name": run_name, "config": dict(config or {})}
        if mode is not None:
            kwargs["mode"] = mode
        if tags:
            kwargs["tags"] = list(tags)
        if group:
            kwargs["group"] = group
        if dir_:
            kwargs["dir"] = dir_
        self._run = wandb.init(**kwargs)
        self._wandb = wandb

    def log(self, data: Mapping[str, Any], step: int | None = None) -> None:
        if not self.enabled:
            return
        if step is None:
            self._wandb.log(dict(data))
        else:
            self._wandb.log(dict(data), step=step)

    def summary(self, data: Mapping[str, Any]) -> None:
        if not self.enabled or self._run is None:
            return
        for k, v in data.items():
            self._wandb.run.summary[k] = v

    def save_file(self, path: str) -> None:
        if not self.enabled:
            return
        self._wandb.save(path)

    def finish(self) -> None:
        if not self.enabled or self._run is None:
            return
        self._wandb.finish()


def from_cfg(cfg: Mapping[str, Any], run_name: str, log_dir: str | None = None) -> WandbLogger:
    wcfg = cfg.get("wandb") or {}
    enabled = bool(wcfg.get("enabled", True))
    return WandbLogger(
        enabled=enabled,
        project=wcfg.get("project", "fracture"),
        run_name=run_name,
        mode=wcfg.get("mode"),  # None | "online" | "offline" | "disabled"
        config=cfg,
        tags=wcfg.get("tags"),
        group=wcfg.get("group"),
        dir_=log_dir,
    )
