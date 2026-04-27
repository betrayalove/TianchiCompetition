from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MetricResult:
    precision: float
    passed_posts: int
    total_posts: int


def competition_precision(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    sign_mode: str = "indicator",
    use_abs_deviation: bool = True,
    cap_count_at: int = 100,
) -> MetricResult:
    true_arr = y_true[["forward_count", "comment_count", "like_count"]].to_numpy(dtype=np.float64)
    pred_arr = y_pred[["forward_count", "comment_count", "like_count"]].to_numpy(dtype=np.float64)
    deviation = (pred_arr - true_arr) / (true_arr + 5.0)
    if use_abs_deviation:
        deviation = np.abs(deviation)
    precision_i = 1.0 - 0.5 * deviation[:, 0] - 0.25 * deviation[:, 1] - 0.25 * deviation[:, 2]

    if sign_mode == "indicator":
        signed = (precision_i > 0.8).astype(np.float64)
    elif sign_mode == "math_sign":
        signed = np.sign(precision_i - 0.8)
    else:
        raise ValueError(f"Unsupported sign_mode: {sign_mode}")

    counts = np.minimum(true_arr.sum(axis=1), cap_count_at) + 1.0
    score = float(np.sum(counts * signed) / np.sum(counts))
    return MetricResult(
        precision=score,
        passed_posts=int(np.sum(precision_i > 0.8)),
        total_posts=int(true_arr.shape[0]),
    )
