import argparse
import logging
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import shap
import sqlalchemy as sa
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


LOG = logging.getLogger("f1_models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train F1 gradient boosting ensemble models.")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--optuna-trials", type=int, default=25)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def get_engine(database_url: str) -> sa.Engine:
    if not database_url:
        raise ValueError("DATABASE_URL is required, e.g. postgresql+psycopg2://user:pass@host/db")
    return sa.create_engine(database_url, future=True)


def load_feature_matrix(engine: sa.Engine) -> pd.DataFrame:
    query = sa.text(
        """
        SELECT
            rr.race_id,
            rr.driver_id,
            rr.team_id,
            rr.finishing_position,
            rr.race_time,
            rr.status,
            r.season,
            r.round,
            df.*, tf.*, sf.*
        FROM race_results rr
        JOIN races r ON rr.race_id = r.race_id
        LEFT JOIN driver_features df ON rr.race_id = df.race_id AND rr.driver_id = df.driver_id
        LEFT JOIN team_features tf ON rr.race_id = tf.race_id AND rr.team_id = tf.team_id
        LEFT JOIN situational_features sf ON rr.race_id = sf.race_id AND rr.driver_id = sf.driver_id
        """
    )
    df = pd.read_sql(query, engine)
    df = df.sort_values(["season", "round"]).reset_index(drop=True)
    df["order_index"] = range(len(df))
    return df


def prepare_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    winner_time = df.groupby("race_id")["race_time"].transform("min")
    df["race_time_delta"] = df["race_time"] - winner_time
    df["dnf"] = df["status"].fillna("").str.contains("DNF|DNS|DSQ|Ret", case=False)
    df["podium_class"] = pd.cut(
        df["finishing_position"],
        bins=[0, 1, 2, 3, 10, 20],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True,
    ).astype("Int64")
    return df


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    feature_cols = [
        col
        for col in df.columns
        if col
        not in {
            "race_id",
            "driver_id",
            "team_id",
            "finishing_position",
            "race_time",
            "race_time_delta",
            "status",
            "season",
            "round",
            "order_index",
            "dnf",
            "podium_class",
        }
    ]
    features = df[feature_cols].select_dtypes(include=["number"]).fillna(0)
    return features, feature_cols


def time_series_cv(
    X: pd.DataFrame, y: pd.Series, n_splits: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(X, y))


def tune_xgb_position(
    X: pd.DataFrame,
    y: pd.Series,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    trials: int,
) -> Dict:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "reg:squarederror",
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "max_depth": trial.suggest_int("max_depth", 4, 9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "gamma": trial.suggest_float("gamma", 0.0, 0.2),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.3),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 1.5),
            "random_state": 42,
        }
        scores = []
        for train_idx, test_idx in splits:
            model = xgb.XGBRegressor(**params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[test_idx])
            scores.append(mean_squared_error(y.iloc[test_idx], preds))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials)
    return study.best_params


def train_xgb_position(X: pd.DataFrame, y: pd.Series, params: Dict) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model


def tune_lgb_time(
    X: pd.DataFrame,
    y: pd.Series,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    trials: int,
) -> Dict:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.9),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.9),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "n_estimators": trial.suggest_int("n_estimators", 150, 400),
        }
        scores = []
        for train_idx, test_idx in splits:
            model = lgb.LGBMRegressor(**params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[test_idx])
            scores.append(mean_absolute_error(y.iloc[test_idx], preds))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials)
    return study.best_params


def train_lgb_time(X: pd.DataFrame, y: pd.Series, params: Dict) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)
    return model


def train_xgb_dnf(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    params = {
        "objective": "binary:logistic",
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "scale_pos_weight": 10,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X, y)
    return model


def train_xgb_podium(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    params = {
        "objective": "multi:softprob",
        "num_class": 5,
        "n_estimators": 300,
        "max_depth": 6,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X, y)
    return model


def evaluate_models(
    X: pd.DataFrame,
    y_position: pd.Series,
    y_time: pd.Series,
    y_dnf: pd.Series,
    y_podium: pd.Series,
    model_dir: str,
    n_splits: int,
    optuna_trials: int,
) -> None:
    os.makedirs(model_dir, exist_ok=True)

    splits = time_series_cv(X, y_position, n_splits=n_splits)

    xgb_params = tune_xgb_position(X, y_position, splits, optuna_trials)
    lgb_params = tune_lgb_time(X, y_time, splits, optuna_trials)

    position_model = train_xgb_position(X, y_position, xgb_params)
    time_model = train_lgb_time(X, y_time, lgb_params)
    dnf_model = train_xgb_dnf(X, y_dnf)
    podium_model = train_xgb_podium(X, y_podium)

    joblib.dump(position_model, os.path.join(model_dir, "xgb_position.joblib"))
    joblib.dump(time_model, os.path.join(model_dir, "lgb_time.joblib"))
    joblib.dump(dnf_model, os.path.join(model_dir, "xgb_dnf.joblib"))
    joblib.dump(podium_model, os.path.join(model_dir, "xgb_podium.joblib"))

    position_preds = position_model.predict(X)
    time_preds = time_model.predict(X)
    dnf_preds = dnf_model.predict_proba(X)[:, 1]
    podium_preds = podium_model.predict_proba(X)

    LOG.info("Position RMSE: %.3f", mean_squared_error(y_position, position_preds, squared=False))
    LOG.info("Time MAE: %.3f", mean_absolute_error(y_time, time_preds))
    LOG.info("DNF ROC-AUC: %.3f", roc_auc_score(y_dnf, dnf_preds))
    LOG.info("Podium Class MAE: %.3f", mean_absolute_error(y_podium, podium_preds.argmax(axis=1)))

    explain_model(position_model, X, model_dir, "xgb_position")
    explain_model(time_model, X, model_dir, "lgb_time")
    explain_model(dnf_model, X, model_dir, "xgb_dnf")
    explain_model(podium_model, X, model_dir, "xgb_podium")


def explain_model(model, X: pd.DataFrame, model_dir: str, name: str) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    summary_path = os.path.join(model_dir, f"{name}_shap_summary.npy")
    np.save(summary_path, shap_values)


def main() -> None:
    setup_logging()
    args = parse_args()
    engine = get_engine(args.database_url)
    df = load_feature_matrix(engine)
    df = prepare_targets(df)
    features, _ = select_features(df)

    evaluate_models(
        features,
        df["finishing_position"],
        df["race_time_delta"],
        df["dnf"].astype(int),
        df["podium_class"].astype(int),
        args.model_dir,
        args.n_splits,
        args.optuna_trials,
    )


if __name__ == "__main__":
    main()
