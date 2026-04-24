"""Evaluation harnesses, metrics, and result aggregation.

Lazy: `evaluate_from_cfg` pulls torch; access triggers the import.
"""
from .registry import write_row, dump_metrics

__all__ = [
    "masked_mse",
    "masked_mae",
    "masked_relative_l2",
    "damage_iou",
    "irreversibility_violation",
    "nonphysical_rate",
    "summarize",
    "merge_prefix",
    "write_row",
    "dump_metrics",
    "evaluate_from_cfg",
]


def __getattr__(name):
    if name == "evaluate_from_cfg":
        from .evaluator import evaluate_from_cfg
        return evaluate_from_cfg
    if name in {
        "masked_mse", "masked_mae", "masked_relative_l2",
        "damage_iou", "irreversibility_violation", "nonphysical_rate",
        "summarize", "merge_prefix",
    }:
        from . import metrics
        return getattr(metrics, name)
    raise AttributeError(name)
