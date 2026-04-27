from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

from .data import TARGET_COLUMNS


NUMERIC_FEATURE_COLUMNS = [
    "hour",
    "day",
    "month",
    "dayofweek",
    "is_weekend",
    "content_len",
    "digit_count",
    "url_count",
    "mention_count",
    "hashtag_count",
    "bracket_emoji_count",
    "exclam_count",
    "question_count",
    "stock_tag_count",
    "contains_url",
    "contains_mention",
    "contains_hashtag",
    "uid_post_count",
    "uid_forward_mean",
    "uid_comment_mean",
    "uid_like_mean",
    "uid_total_mean",
    "uid_forward_median",
    "uid_comment_median",
    "uid_like_median",
    "uid_total_median",
    "uid_forward_max",
    "uid_comment_max",
    "uid_like_max",
    "uid_nonzero_rate",
]


def build_user_stats(history_df: pd.DataFrame) -> pd.DataFrame:
    enriched = history_df.copy()
    enriched["total_interactions"] = enriched[TARGET_COLUMNS].sum(axis=1)
    stats = (
        enriched.groupby("uid", observed=True)
        .agg(
            uid_post_count=("mid", "size"),
            uid_forward_mean=("forward_count", "mean"),
            uid_comment_mean=("comment_count", "mean"),
            uid_like_mean=("like_count", "mean"),
            uid_total_mean=("total_interactions", "mean"),
            uid_forward_median=("forward_count", "median"),
            uid_comment_median=("comment_count", "median"),
            uid_like_median=("like_count", "median"),
            uid_total_median=("total_interactions", "median"),
            uid_forward_max=("forward_count", "max"),
            uid_comment_max=("comment_count", "max"),
            uid_like_max=("like_count", "max"),
            uid_nonzero_rate=("total_interactions", lambda x: float((x > 0).mean())),
        )
        .reset_index()
    )
    return stats


def attach_engineered_features(df: pd.DataFrame, user_stats: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    text = out["content"].astype("string")
    out["content_len"] = text.str.len().fillna(0).astype("float32")
    out["digit_count"] = text.str.count(r"\d").fillna(0).astype("float32")
    out["url_count"] = text.str.count(r"http[s]?://|http://|https://|t\.cn/").fillna(0).astype("float32")
    out["mention_count"] = text.str.count(r"@").fillna(0).astype("float32")
    out["hashtag_count"] = text.str.count(r"#").fillna(0).astype("float32")
    out["bracket_emoji_count"] = text.str.count(r"\[[^\]]+\]").fillna(0).astype("float32")
    out["exclam_count"] = text.str.count(r"!|！").fillna(0).astype("float32")
    out["question_count"] = text.str.count(r"\?|？").fillna(0).astype("float32")
    out["stock_tag_count"] = text.str.count(r"sz\d+|sh\d+|股票|理财|红包").fillna(0).astype("float32")
    out["contains_url"] = (out["url_count"] > 0).astype("float32")
    out["contains_mention"] = (out["mention_count"] > 0).astype("float32")
    out["contains_hashtag"] = (out["hashtag_count"] > 0).astype("float32")
    out = out.merge(user_stats, on="uid", how="left")
    for column in NUMERIC_FEATURE_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0
    out[NUMERIC_FEATURE_COLUMNS] = out[NUMERIC_FEATURE_COLUMNS].fillna(0.0).astype("float32")
    return out


def build_sparse_design_matrix(
    df: pd.DataFrame,
    vectorizer: HashingVectorizer | None = None,
) -> tuple[sparse.csr_matrix, HashingVectorizer]:
    if vectorizer is None:
        vectorizer = HashingVectorizer(
            n_features=2**16,
            alternate_sign=False,
            analyzer="char",
            ngram_range=(2, 3),
            norm="l2",
            lowercase=False,
        )
    text_matrix = vectorizer.transform(df["content"].astype(str).tolist())
    numeric_matrix = sparse.csr_matrix(df[NUMERIC_FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    design = sparse.hstack([numeric_matrix, text_matrix], format="csr")
    return design, vectorizer


def engagement_weights(df: pd.DataFrame) -> np.ndarray:
    total = df[TARGET_COLUMNS].sum(axis=1).to_numpy(dtype=np.float32)
    return np.minimum(total, 100.0) + 1.0


def clip_and_round_predictions(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = np.maximum(arr, 0.0)
    return np.rint(arr).astype(np.int32)
