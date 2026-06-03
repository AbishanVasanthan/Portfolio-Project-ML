"""
train_sku.py — Train 6 XGBoost models for per-SKU demand forecasting.

Mirrors train.py but operates on the SKU feature panel
(24 depots × 6 SKUs × 850 weeks ≈ 122,400 rows).

Model names: cement_sku_forecaster_h1 … h6
"""

import logging
import os
import warnings
from datetime import datetime

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)

SKU_REGISTRY = "cement_sku_forecaster"
TARGET        = "target_demand"
HORIZONS      = [1, 2, 3, 4, 5, 6]

EXCLUDE_COLS = {
    "week_start", "depot", "sku_code", "depot_id", "sku_id",
    "demand_tonnes", "sales_tonnes", "data_source",
    "mix_ratio",
}


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in EXCLUDE_COLS
        and not c.startswith("Unnamed")
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().any()
    ]


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _get_production_mape(model_name: str) -> float | None:
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            return None
        run = mlflow.get_run(versions[0].run_id)
        return float(run.data.metrics.get("val_mape", float("inf")))
    except Exception:
        return None


def train_sku_horizons(cfg: dict, retrain_id: int | None = None) -> dict:
    """
    Train one XGBoost model per forecast horizon on the SKU feature panel.
    Returns dict with overall_mape, per_horizon_mape, promoted flag.
    """
    from src.features.build_sku_features import rebuild_sku_features

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    tracking_user = os.getenv("MLFLOW_TRACKING_USERNAME")
    tracking_pass = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if tracking_user:
        os.environ["MLFLOW_TRACKING_USERNAME"] = tracking_user
    if tracking_pass:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = tracking_pass

    mlflow.set_experiment("cement_sku_demand_forecaster")

    df = rebuild_sku_features(cfg)
    if df.empty:
        raise RuntimeError("[TRAIN_SKU] Empty feature panel — run --mode setup_skus first")

    logger.info("[TRAIN_SKU] Panel: %d rows | %d depots | %d SKUs",
                len(df), df["depot"].nunique(), df["sku_code"].nunique())

    cv_train = cfg["model"]["cv_train_weeks"]
    cv_val   = cfg["model"]["cv_val_weeks"]
    cv_step  = cfg["model"]["cv_step_weeks"]
    cv_min   = cfg["model"]["cv_min_folds"]
    n_trials = cfg["model"]["optuna_trials"]
    xgb_cfg  = cfg["model"]["xgb"]

    unique_weeks = sorted(df["week_start"].unique())
    feature_cols = _feature_cols(df)
    logger.info("[TRAIN_SKU] Features (%d): %s", len(feature_cols), feature_cols)

    horizon_mapes: dict[int, float] = {}
    promoted_any = False

    for horizon in HORIZONS:
        logger.info("[TRAIN_SKU] ─── Horizon %d ───", horizon)

        # Build target: shift demand by horizon within (depot, SKU)
        frame = df.copy()
        frame[TARGET] = frame.groupby(["depot", "sku_code"])["demand_tonnes"].shift(-horizon)
        valid = frame.dropna(subset=[TARGET]).copy()
        valid[feature_cols] = valid[feature_cols].fillna(0.0)

        # Build rolling CV splits
        splits: list[tuple[pd.Series, pd.Series]] = []
        start_idx = cv_train
        while start_idx + cv_val <= len(unique_weeks):
            train_weeks = unique_weeks[start_idx - cv_train: start_idx]
            val_weeks   = unique_weeks[start_idx: start_idx + cv_val]
            tr_mask = valid["week_start"].isin(train_weeks)
            va_mask = valid["week_start"].isin(val_weeks)
            if tr_mask.sum() > 0 and va_mask.sum() > 0:
                splits.append((tr_mask, va_mask))
            start_idx += cv_step

        if len(splits) < cv_min:
            logger.warning("[TRAIN_SKU] h%d: only %d CV folds, skipping", horizon, len(splits))
            continue

        logger.info("[TRAIN_SKU] h%d: %d CV folds, %d valid rows", horizon, len(splits), len(valid))

        # Optuna objective
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", *xgb_cfg["n_estimators"]),
                "max_depth":         trial.suggest_int("max_depth", *xgb_cfg["max_depth"]),
                "learning_rate":     trial.suggest_float("learning_rate", *xgb_cfg["learning_rate"], log=True),
                "subsample":         trial.suggest_float("subsample", *xgb_cfg["subsample"]),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", *xgb_cfg["colsample_bytree"]),
                "min_child_weight":  trial.suggest_int("min_child_weight", *xgb_cfg["min_child_weight"]),
                "tree_method": "hist",
                "random_state": 42,
            }
            fold_mapes = []
            for tr_mask, va_mask in splits:
                model = xgb.XGBRegressor(**params, verbosity=0)
                model.fit(valid.loc[tr_mask, feature_cols], valid.loc[tr_mask, TARGET])
                preds = np.clip(model.predict(valid.loc[va_mask, feature_cols]), 0, None)
                fold_mapes.append(_mape(valid.loc[va_mask, TARGET].to_numpy(float), preds))
            return float(np.mean(fold_mapes))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = {**study.best_params, "tree_method": "hist", "random_state": 42}

        # Final model on full dataset
        final_model = xgb.XGBRegressor(**best_params, verbosity=0)
        final_model.fit(valid[feature_cols], valid[TARGET])

        # Validation MAPE on last fold
        last_tr, last_va = splits[-1]
        val_preds = np.clip(final_model.predict(valid.loc[last_va, feature_cols]), 0, None)
        val_mape  = _mape(valid.loc[last_va, TARGET].to_numpy(float), val_preds)
        horizon_mapes[horizon] = val_mape
        logger.info("[TRAIN_SKU] h%d val_mape=%.2f%%", horizon, val_mape)

        # MLflow logging + registration
        model_name = f"{SKU_REGISTRY}_h{horizon}"
        prev_mape  = _get_production_mape(model_name)
        with mlflow.start_run(run_name=f"sku_xgb_h{horizon}") as run:
            mlflow.log_params(best_params)
            mlflow.log_metrics({"val_mape": val_mape, "cv_folds": len(splits)})
            if retrain_id:
                mlflow.set_tag("retrain_id", str(retrain_id))
            mlflow.xgboost.log_model(final_model, name=model_name,
                                     registered_model_name=model_name)

        # Promote if better than previous Production
        if prev_mape is None or val_mape <= prev_mape:
            client = MlflowClient()
            versions = client.get_latest_versions(model_name, stages=["None", "Staging"])
            if versions:
                client.transition_model_version_stage(
                    name=model_name,
                    version=versions[-1].version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                logger.info("[TRAIN_SKU] Promoted %s v%s to Production (%.2f%% vs %.2f%%)",
                            model_name, versions[-1].version, val_mape,
                            prev_mape if prev_mape else float("inf"))
                promoted_any = True

    overall_mape = float(np.mean(list(horizon_mapes.values()))) if horizon_mapes else 0.0
    logger.info("[TRAIN_SKU] Overall avg MAPE: %.2f%%", overall_mape)

    return {
        "overall_mape":    overall_mape,
        "horizon_mapes":   horizon_mapes,
        "promoted":        promoted_any,
    }
