from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor

from src.weibo_solution.data import TARGET_COLUMNS, load_predict_data, load_train_data
from src.weibo_solution.features import (
    attach_engineered_features,
    build_sparse_design_matrix,
    build_user_stats,
    clip_and_round_predictions,
    engagement_weights,
)
from src.weibo_solution.metrics import competition_precision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline model for Weibo interaction prediction.")
    parser.add_argument("--train-path", default="Weibo Data/weibo_train_data.txt")
    parser.add_argument("--predict-path", default="Weibo Data/weibo_predict_data.txt")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--validation-start", default="2015-07-01")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-predict-rows", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--enable-text-model", action="store_true")
    return parser.parse_args()


def build_regressor(random_state: int) -> SGDRegressor:
    return SGDRegressor(
        loss="huber",
        penalty="l2",
        alpha=0.0005,
        max_iter=15,
        tol=1e-3,
        random_state=random_state,
        learning_rate="invscaling",
        eta0=0.01,
        power_t=0.25,
        average=True,
        early_stopping=False,
    )


def inverse_target_transform(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, np.log1p(50000.0))
    return np.expm1(clipped)


def fit_and_predict(train_x, train_y, valid_x, test_x, weights, random_state: int):
    valid_preds = {}
    test_preds = {}
    for idx, target in enumerate(TARGET_COLUMNS):
        model = build_regressor(random_state + idx)
        transformed = np.log1p(train_y[target].to_numpy(dtype=np.float64))
        model.fit(train_x, transformed, sample_weight=weights)
        valid_preds[target] = clip_and_round_predictions(inverse_target_transform(model.predict(valid_x)))
        test_preds[target] = clip_and_round_predictions(inverse_target_transform(model.predict(test_x)))
    return pd.DataFrame(valid_preds), pd.DataFrame(test_preds)


def history_baseline_from_features(df: pd.DataFrame) -> pd.DataFrame:
    baseline = pd.DataFrame(
        {
            "forward_count": df["uid_forward_mean"].round().clip(lower=0).astype("int32"),
            "comment_count": df["uid_comment_mean"].round().clip(lower=0).astype("int32"),
            "like_count": df["uid_like_mean"].round().clip(lower=0).astype("int32"),
        }
    )
    return baseline


def blend_predictions(left: pd.DataFrame, right: pd.DataFrame, right_weight: float) -> pd.DataFrame:
    out = pd.DataFrame()
    for column in TARGET_COLUMNS:
        blended = (1.0 - right_weight) * left[column].to_numpy(dtype=np.float64) + right_weight * right[column].to_numpy(
            dtype=np.float64
        )
        out[column] = clip_and_round_predictions(blended)
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_train_data(args.train_path, max_rows=args.max_train_rows)
    predict_df = load_predict_data(args.predict_path, max_rows=args.max_predict_rows)

    validation_start = pd.Timestamp(args.validation_start)
    fit_df = train_df.loc[train_df["time"] < validation_start].copy()
    valid_df = train_df.loc[train_df["time"] >= validation_start].copy()
    if fit_df.empty or valid_df.empty:
        raise ValueError("Time split produced an empty train or validation fold. Adjust --validation-start.")

    user_stats = build_user_stats(fit_df)
    fit_features = attach_engineered_features(fit_df, user_stats)
    valid_features = attach_engineered_features(valid_df, user_stats)
    predict_features = attach_engineered_features(predict_df, user_stats)

    valid_pred = history_baseline_from_features(valid_features)
    test_pred = history_baseline_from_features(predict_features)
    selected_model = "user_history_mean"
    all_metrics = {}

    history_indicator = competition_precision(valid_df[TARGET_COLUMNS], valid_pred, sign_mode="indicator")
    history_math = competition_precision(valid_df[TARGET_COLUMNS], valid_pred, sign_mode="math_sign")
    all_metrics["user_history_mean"] = {
        "indicator_precision": history_indicator.precision,
        "math_sign_precision": history_math.precision,
    }

    if args.enable_text_model:
        train_x, vectorizer = build_sparse_design_matrix(fit_features)
        valid_x, _ = build_sparse_design_matrix(valid_features, vectorizer=vectorizer)
        test_x, _ = build_sparse_design_matrix(predict_features, vectorizer=vectorizer)
        weights = engagement_weights(fit_df)

        text_valid_pred, text_test_pred = fit_and_predict(
            train_x=train_x,
            train_y=fit_df[TARGET_COLUMNS],
            valid_x=valid_x,
            test_x=test_x,
            weights=weights,
            random_state=args.random_state,
        )

        blended_valid = blend_predictions(valid_pred, text_valid_pred, right_weight=0.35)
        blended_test = blend_predictions(test_pred, text_test_pred, right_weight=0.35)
        blend_indicator = competition_precision(valid_df[TARGET_COLUMNS], blended_valid, sign_mode="indicator")
        blend_math = competition_precision(valid_df[TARGET_COLUMNS], blended_valid, sign_mode="math_sign")
        all_metrics["history_plus_text_blend"] = {
            "indicator_precision": blend_indicator.precision,
            "math_sign_precision": blend_math.precision,
        }
        if blend_indicator.precision >= history_indicator.precision:
            valid_pred = blended_valid
            test_pred = blended_test
            selected_model = "history_plus_text_blend"

    indicator_metric = competition_precision(valid_df[TARGET_COLUMNS], valid_pred, sign_mode="indicator")
    math_sign_metric = competition_precision(valid_df[TARGET_COLUMNS], valid_pred, sign_mode="math_sign")

    submission = predict_df[["uid", "mid"]].copy()
    for column in TARGET_COLUMNS:
        submission[column] = test_pred[column].astype("int32")

    submission_path = output_dir / "submission_baseline.txt"
    submission.to_csv(submission_path, sep="\t", index=False, header=False)

    report = {
        "train_rows": int(len(fit_df)),
        "validation_rows": int(len(valid_df)),
        "predict_rows": int(len(predict_df)),
        "validation_start": args.validation_start,
        "selected_model": selected_model,
        "indicator_precision": indicator_metric.precision,
        "indicator_passed_posts": indicator_metric.passed_posts,
        "math_sign_precision": math_sign_metric.precision,
        "math_sign_passed_posts": math_sign_metric.passed_posts,
        "candidate_metrics": all_metrics,
        "notes": [
            "The local metric uses abs((pred-real)/(real+5)) based on competition writeups.",
            "The indicator version is closer to archived descriptions where sign behaves like 1/0.",
            "The math_sign version is included because the supplied formula defines sign as -1/0/1.",
            "The default baseline predicts per-user mean interactions from historical posts.",
            "The optional text model is blended in only if it improves the validation score.",
        ],
    }
    report_path = output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Submission saved to: {submission_path}")


if __name__ == "__main__":
    main()
