from __future__ import annotations

from pathlib import Path

import pandas as pd


TRAIN_COLUMNS = [
    "uid",
    "mid",
    "time",
    "forward_count",
    "comment_count",
    "like_count",
    "content",
]

PREDICT_COLUMNS = [
    "uid",
    "mid",
    "time",
    "content",
]

TARGET_COLUMNS = ["forward_count", "comment_count", "like_count"]


def load_train_data(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    paths = _resolve_train_paths(path)
    remaining_rows = max_rows
    frames: list[pd.DataFrame] = []
    for current_path in paths:
        current_nrows = remaining_rows if remaining_rows is not None else None
        df = pd.read_csv(
            current_path,
            sep="\t",
            header=None,
            names=TRAIN_COLUMNS,
            nrows=current_nrows,
            dtype={
                "uid": "string",
                "mid": "string",
                "time": "string",
                "forward_count": "int32",
                "comment_count": "int32",
                "like_count": "int32",
                "content": "string",
            },
        )
        frames.append(df)
        if remaining_rows is not None:
            remaining_rows -= len(df)
            if remaining_rows <= 0:
                break
    combined = pd.concat(frames, ignore_index=True)
    return _postprocess_common(combined)


def load_predict_data(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=PREDICT_COLUMNS,
        nrows=max_rows,
        dtype={
            "uid": "string",
            "mid": "string",
            "time": "string",
            "content": "string",
        },
    )
    return _postprocess_common(df)


def _resolve_train_paths(path: str | Path) -> list[Path]:
    candidate = Path(path)
    if candidate.exists():
        return [candidate]
    part_paths = sorted(candidate.parent.glob(f"{candidate.name}.part*"))
    if not part_paths:
        raise FileNotFoundError(f"Train data not found at {candidate} and no split parts matched.")
    return part_paths


def _postprocess_common(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["content"] = out["content"].fillna("")
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out["date"] = out["time"].dt.date
    out["hour"] = out["time"].dt.hour.astype("int8")
    out["day"] = out["time"].dt.day.astype("int8")
    out["month"] = out["time"].dt.month.astype("int8")
    out["dayofweek"] = out["time"].dt.dayofweek.astype("int8")
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype("int8")
    return out
