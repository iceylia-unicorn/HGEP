# src/gpbench/utils/early_stop.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import torch


@dataclass
class EarlyStopConfig:
    patience: int = 7
    min_delta: float = 0.0       # improvement threshold
    mode: str = "min"            # "min" for loss, "max" for acc
    save_best: bool = True
    save_path: str = "checkpoints/best.pt"


class EarlyStopping:
    def __init__(self, cfg: EarlyStopConfig):
        if cfg.mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.cfg = cfg
        self.best: Optional[float] = None
        self.bad_epochs: int = 0
        self.best_epoch: int = -1

    def _is_improved(self, metric: float) -> bool:
        if self.best is None:
            return True
        if self.cfg.mode == "min":
            return metric < (self.best - self.cfg.min_delta)
        else:
            return metric > (self.best + self.cfg.min_delta)

    def step(
        self,
        metric: float,
        epoch: int,
        model: torch.nn.Module,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Returns True if should stop.
        """
        if self._is_improved(metric):
            self.best = float(metric)
            self.bad_epochs = 0
            self.best_epoch = int(epoch)

            if self.cfg.save_best:
                path = Path(self.cfg.save_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = {"model_state": model.state_dict(), "best_metric": self.best, "best_epoch": self.best_epoch}
                if extra_state:
                    payload.update(extra_state)
                torch.save(payload, str(path))
        else:
            self.bad_epochs += 1

        return self.bad_epochs >= self.cfg.patience
