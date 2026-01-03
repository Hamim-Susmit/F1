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
import tensorflow as tf
from tensorflow import keras
from keras import layers
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


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
    tf.get_logger().setLevel("ERROR")


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
    df["winner"] = df["finishing_position"] == 1
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
            "winner",
            "podium_class",
        }
    ]
    features = df[feature_cols].select_dtypes(include=["number"]).fillna(0)
    return features, feature_cols


def create_cv_folds(seasons: pd.Series) -> List[Dict[str, str]]:
    folds = [
        {"train": "2010-2017", "val": "2018", "description": "Early season baseline"},
        {"train": "2010-2019", "val": "2020", "description": "Pre-COVID era"},
        {"train": "2010-2021", "val": "2022", "description": "Before new regs"},
        {"train": "2010-2022", "val": "2023", "description": "Post new regs"},
        {"train": "2010-2023", "val": "2024", "description": "Most recent"},
        {"train": "2010-2024", "test": "2025", "description": "Final test set"},
    ]
    available = set(seasons.unique())
    valid_folds = []
    for fold in folds:
        train_end = int(fold["train"].split("-")[1])
        train_years = list(range(2010, train_end + 1))
        if not set(train_years).issubset(available):
            continue
        val_year = int(fold.get("val", fold.get("test")))
        if val_year not in available:
            continue
        valid_folds.append(fold)
    return valid_folds


def build_fold_indices(
    df: pd.DataFrame, folds: List[Dict[str, str]]
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    fold_indices = []
    for fold in folds:
        train_end = int(fold["train"].split("-")[1])
        train_years = list(range(2010, train_end + 1))
        val_year = int(fold.get("val", fold.get("test")))
        train_mask = df["season"].isin(train_years)
        val_mask = df["season"] == val_year
        fold_indices.append((df.index[train_mask].to_numpy(), df.index[val_mask].to_numpy(), fold["description"]))
    return fold_indices
def time_series_cv(
    X: pd.DataFrame, y: pd.Series, n_splits: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(X, y))


def tune_xgb_position(
    X: pd.DataFrame,
    y: pd.Series,
    splits: List[Tuple[np.ndarray, np.ndarray, str]],
    splits: List[Tuple[np.ndarray, np.ndarray]],
    trials: int,
) -> Dict:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "reg:squarederror",
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "random_state": 42,
        }
        scores = []
        for train_idx, test_idx, _ in splits:
            model = xgb.XGBRegressor(**params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[test_idx])
            scores.append(mean_absolute_error(y.iloc[test_idx], preds))
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
    splits: List[Tuple[np.ndarray, np.ndarray, str]],
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
        for train_idx, test_idx, _ in splits:
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


def train_lgb_position(X: pd.DataFrame, y: pd.Series, params: Dict) -> lgb.LGBMRegressor:
    lgb_params = dict(params)
    lgb_params.update({"objective": "quantile", "alpha": 0.5})
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X, y)
    return model


def train_lgb_position_quantile(
    X: pd.DataFrame, y: pd.Series, params: Dict, alpha: float
) -> lgb.LGBMRegressor:
    lgb_params = dict(params)
    lgb_params.update({"objective": "quantile", "alpha": alpha})
    model = lgb.LGBMRegressor(**lgb_params)
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)
    return model


def train_xgb_time(X: pd.DataFrame, y: pd.Series, params: Dict) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model


def resample_for_dnf(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    over = SMOTE(sampling_strategy=0.3, random_state=42)
    under = RandomUnderSampler(sampling_strategy=0.6, random_state=42)
    pipeline = Pipeline([("over", over), ("under", under)])
    X_resampled, y_resampled = pipeline.fit_resample(X, y)
    return X_resampled, y_resampled


def train_xgb_dnf(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    class_ratio = (y == 0).sum() / max((y == 1).sum(), 1)
def train_xgb_dnf(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    params = {
        "objective": "binary:logistic",
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "scale_pos_weight": class_ratio,
    }
    X_resampled, y_resampled = resample_for_dnf(X, y)
    model = xgb.XGBClassifier(**params)
    model.fit(X_resampled, y_resampled)
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
    classes = np.array(sorted(y.unique()))
    weights = compute_class_weight("balanced", classes=classes, y=y)
    class_weight = dict(zip(classes, weights))
    sample_weight = y.map(class_weight).to_numpy()
    model = xgb.XGBClassifier(**params)
    model.fit(X, y, sample_weight=sample_weight)
    model = xgb.XGBClassifier(**params)
    model.fit(X, y)
    return model


def build_f1_neural_network(input_dim: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu", kernel_regularizer="l2"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu", kernel_regularizer="l2"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu", kernel_regularizer="l2"),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae", "mape"],
    )
    return model


def build_position_distribution_network(input_dim: int, num_classes: int = 20) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu", kernel_regularizer="l2"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu", kernel_regularizer="l2"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_neural_network(
    X: pd.DataFrame,
    y: pd.Series,
    model_dir: str,
    name: str,
) -> keras.Model:
    tf.random.set_seed(42)
    model = build_f1_neural_network(X.shape[1])
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
    )
    model.fit(
        X,
        y,
        validation_split=0.15,
        epochs=200,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr],
        verbose=0,
    )
    model.save(os.path.join(model_dir, f"{name}.keras"))
    return model


def train_position_distribution_network(
    X: pd.DataFrame,
    y: pd.Series,
    model_dir: str,
) -> keras.Model:
    tf.random.set_seed(42)
    y_onehot = keras.utils.to_categorical(y.clip(lower=1) - 1, num_classes=20)
    model = build_position_distribution_network(X.shape[1])
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
    )
    model.fit(
        X,
        y_onehot,
        validation_split=0.15,
        epochs=200,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr],
        verbose=0,
    )
    model.save(os.path.join(model_dir, "nn_position_distribution.keras"))
    return model


def train_rf_regressor(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    params = {
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1,
    }
    model = RandomForestRegressor(**params)
    model.fit(X, y)
    return model


def train_rf_classifier(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    params = {
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced",
    }
    model = RandomForestClassifier(**params)
    model.fit(X, y)
    return model


def evaluate_models(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y_position: pd.Series,
    y_time: pd.Series,
    y_dnf: pd.Series,
    y_podium: pd.Series,
    y_winner: pd.Series,
    model_dir: str,
    n_splits: int,
    optuna_trials: int,
) -> None:
    os.makedirs(model_dir, exist_ok=True)

    folds = create_cv_folds(df["season"])
    fold_indices = build_fold_indices(df, folds)

    xgb_params = tune_xgb_position(X, y_position, fold_indices, optuna_trials)
    lgb_params = tune_lgb_time(X, y_time, fold_indices, optuna_trials)

    evaluate_time_series_folds(
        df,
        X,
        y_position,
        y_time,
        y_winner,
        xgb_params,
        lgb_params,
        fold_indices,
    )
    splits = time_series_cv(X, y_position, n_splits=n_splits)

    xgb_params = tune_xgb_position(X, y_position, splits, optuna_trials)
    lgb_params = tune_lgb_time(X, y_time, splits, optuna_trials)

    position_model = train_xgb_position(X, y_position, xgb_params)
    time_model = train_lgb_time(X, y_time, lgb_params)
    lgb_position_model = train_lgb_position(X, y_position, lgb_params)
    lgb_position_low = train_lgb_position_quantile(X, y_position, lgb_params, 0.1)
    lgb_position_high = train_lgb_position_quantile(X, y_position, lgb_params, 0.9)
    xgb_time_model = train_xgb_time(X, y_time, xgb_params)
    dnf_model = train_xgb_dnf(X, y_dnf)
    podium_model = train_xgb_podium(X, y_podium)
    rf_position_model = train_rf_regressor(X, y_position)
    rf_time_model = train_rf_regressor(X, y_time)
    rf_winner_model = train_rf_classifier(X, y_winner)
    nn_position_model = train_neural_network(X, y_position, model_dir, "nn_position")
    nn_time_model = train_neural_network(X, y_time, model_dir, "nn_time")
    train_position_distribution_network(X, y_position, model_dir)

    joblib.dump(position_model, os.path.join(model_dir, "xgb_position.joblib"))
    joblib.dump(time_model, os.path.join(model_dir, "lgb_time.joblib"))
    joblib.dump(lgb_position_model, os.path.join(model_dir, "lgb_position.joblib"))
    joblib.dump(lgb_position_low, os.path.join(model_dir, "lgb_position_q10.joblib"))
    joblib.dump(lgb_position_high, os.path.join(model_dir, "lgb_position_q90.joblib"))
    joblib.dump(xgb_time_model, os.path.join(model_dir, "xgb_time.joblib"))
    joblib.dump(dnf_model, os.path.join(model_dir, "xgb_dnf.joblib"))
    joblib.dump(podium_model, os.path.join(model_dir, "xgb_podium.joblib"))
    joblib.dump(rf_position_model, os.path.join(model_dir, "rf_position.joblib"))
    joblib.dump(rf_time_model, os.path.join(model_dir, "rf_time.joblib"))
    joblib.dump(rf_winner_model, os.path.join(model_dir, "rf_winner.joblib"))

    position_preds = position_model.predict(X)
    time_preds = time_model.predict(X)
    lgb_position_preds = lgb_position_model.predict(X)
    xgb_time_preds = xgb_time_model.predict(X)
    dnf_preds = dnf_model.predict_proba(X)[:, 1]
    podium_preds = podium_model.predict_proba(X)
    rf_position_preds = rf_position_model.predict(X)
    rf_time_preds = rf_time_model.predict(X)
    rf_winner_preds = rf_winner_model.predict_proba(X)[:, 1]
    nn_position_preds = nn_position_model.predict(X, verbose=0).ravel()
    nn_time_preds = nn_time_model.predict(X, verbose=0).ravel()

    LOG.info("Position RMSE: %.3f", mean_squared_error(y_position, position_preds, squared=False))
    LOG.info("Time MAE: %.3f", mean_absolute_error(y_time, time_preds))
    LOG.info("DNF ROC-AUC: %.3f", roc_auc_score(y_dnf, dnf_preds))
    LOG.info("Podium Class MAE: %.3f", mean_absolute_error(y_podium, podium_preds.argmax(axis=1)))
    LOG.info("RF Position RMSE: %.3f", mean_squared_error(y_position, rf_position_preds, squared=False))
    LOG.info("RF Time MAE: %.3f", mean_absolute_error(y_time, rf_time_preds))
    LOG.info("RF Winner ROC-AUC: %.3f", roc_auc_score(y_winner, rf_winner_preds))
    LOG.info("NN Position RMSE: %.3f", mean_squared_error(y_position, nn_position_preds, squared=False))
    LOG.info("NN Time MAE: %.3f", mean_absolute_error(y_time, nn_time_preds))

    ensemble_position = (
        0.35 * position_preds
        + 0.30 * lgb_position_preds
        + 0.25 * rf_position_preds
        + 0.10 * nn_position_preds
    )
    ensemble_time = (
        0.35 * xgb_time_preds
        + 0.30 * time_preds
        + 0.25 * rf_time_preds
        + 0.10 * nn_time_preds
    )
    LOG.info("Ensemble Position RMSE: %.3f", mean_squared_error(y_position, ensemble_position, squared=False))
    LOG.info("Ensemble Time MAE: %.3f", mean_absolute_error(y_time, ensemble_time))

    explain_model(position_model, X, model_dir, "xgb_position")
    explain_model(time_model, X, model_dir, "lgb_time")
    explain_model(dnf_model, X, model_dir, "xgb_dnf")
    explain_model(podium_model, X, model_dir, "xgb_podium")
    critical_features = [
        "grid_position",
        "recent_form_weighted",
        "circuit_historical_performance",
        "team_performance_score",
        "weather_impact_adjusted_pace",
        "qualifying_gap_to_pole",
        "reliability_score",
    ]
    pd.Series(critical_features).to_csv(
        os.path.join(model_dir, "critical_features.csv"), index=False, header=False
    )
    importance_fold = fold_indices[-1] if fold_indices else None
    if importance_fold:
        train_idx, val_idx, _ = importance_fold
        X_train = X.iloc[train_idx]
        y_train = y_position.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y_position.iloc[val_idx]
        sample_size = min(2000, len(X_train))
        X_sample = X_train.sample(sample_size, random_state=42)
        y_sample = y_train.loc[X_sample.index]
        save_shap_feature_importance(position_model, X_sample, model_dir, "xgb_position")
        run_rfe(position_model, X_sample, y_sample, model_dir, "xgb_position")
        save_permutation_importance(position_model, X_val, y_val, model_dir, "xgb_position")
    save_rf_feature_importance(rf_position_model, X, model_dir, "rf_position")
    save_rf_feature_importance(rf_time_model, X, model_dir, "rf_time")
    save_rf_feature_importance(rf_winner_model, X, model_dir, "rf_winner")


def explain_model(model, X: pd.DataFrame, model_dir: str, name: str) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    summary_path = os.path.join(model_dir, f"{name}_shap_summary.npy")
    np.save(summary_path, shap_values)


def save_shap_feature_importance(
    model,
    X: pd.DataFrame,
    model_dir: str,
    name: str,
    top_n: int = 20,
) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_matrix = np.mean([np.abs(values) for values in shap_values], axis=0)
    else:
        shap_matrix = np.abs(shap_values)
    importance = pd.DataFrame(
        {"feature": X.columns, "importance": shap_matrix.mean(axis=0)}
    ).sort_values("importance", ascending=False)
    importance.head(top_n).to_csv(
        os.path.join(model_dir, f"{name}_shap_top{top_n}.csv"), index=False
    )


def run_rfe(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_dir: str,
    name: str,
    n_features: int = 50,
) -> List[str]:
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    rfe.fit(X_train, y_train)
    selected = [f for f, keep in zip(X_train.columns, rfe.support_) if keep]
    pd.Series(selected).to_csv(
        os.path.join(model_dir, f"{name}_rfe_selected.csv"), index=False, header=False
    )
    return selected


def save_permutation_importance(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_dir: str,
    name: str,
    n_repeats: int = 10,
) -> None:
    result = permutation_importance(
        model, X_val, y_val, n_repeats=n_repeats, random_state=42, n_jobs=-1
    )
    importance = pd.DataFrame(
        {"feature": X_val.columns, "importance": result.importances_mean}
    ).sort_values("importance", ascending=False)
    importance.to_csv(
        os.path.join(model_dir, f"{name}_permutation_importance.csv"), index=False
    )


def save_rf_feature_importance(model, X: pd.DataFrame, model_dir: str, name: str) -> None:
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    importance_path = os.path.join(model_dir, f"{name}_feature_importance.csv")
    importance.to_csv(importance_path, header=["importance"])


def compute_fold_metrics(
    df_fold: pd.DataFrame,
    position_preds: np.ndarray,
    time_preds: np.ndarray,
    winner_probs: np.ndarray,
) -> Dict[str, float]:
    eval_df = df_fold.copy()
    eval_df["pred_position"] = position_preds
    eval_df["pred_time"] = time_preds
    eval_df["winner_prob"] = winner_probs

    winner_hits = []
    top3_hits = []
    top10_hits = []
    spearman_scores = []

    for _, race_df in eval_df.groupby("race_id"):
        actual_winner = race_df.loc[race_df["finishing_position"].idxmin(), "driver_id"]
        predicted_winner = race_df.loc[race_df["winner_prob"].idxmax(), "driver_id"]
        winner_hits.append(actual_winner == predicted_winner)

        actual_top3 = set(race_df.nsmallest(3, "finishing_position")["driver_id"])
        pred_top3 = set(race_df.nsmallest(3, "pred_position")["driver_id"])
        top3_hits.append(len(actual_top3 & pred_top3) / 3)

        actual_top10 = set(race_df.nsmallest(10, "finishing_position")["driver_id"])
        pred_top10 = set(race_df.nsmallest(10, "pred_position")["driver_id"])
        top10_hits.append(len(actual_top10 & pred_top10) / 10)

        spearman = pd.Series(race_df["pred_position"]).corr(
            pd.Series(race_df["finishing_position"]), method="spearman"
        )
        if not np.isnan(spearman):
            spearman_scores.append(spearman)

    return {
        "winner_accuracy": float(np.mean(winner_hits)) if winner_hits else 0.0,
        "top3_accuracy": float(np.mean(top3_hits)) if top3_hits else 0.0,
        "top10_accuracy": float(np.mean(top10_hits)) if top10_hits else 0.0,
        "position_mae": float(mean_absolute_error(eval_df["finishing_position"], eval_df["pred_position"])),
        "time_mae": float(mean_absolute_error(eval_df["race_time_delta"], eval_df["pred_time"])),
        "spearman": float(np.mean(spearman_scores)) if spearman_scores else 0.0,
    }


def evaluate_time_series_folds(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y_position: pd.Series,
    y_time: pd.Series,
    y_winner: pd.Series,
    xgb_params: Dict,
    lgb_params: Dict,
    folds: List[Tuple[np.ndarray, np.ndarray, str]],
) -> None:
    for train_idx, val_idx, label in folds:
        X_train = X.iloc[train_idx]
        y_pos_train = y_position.iloc[train_idx]
        y_time_train = y_time.iloc[train_idx]
        y_win_train = y_winner.iloc[train_idx]

        position_model = train_xgb_position(X_train, y_pos_train, xgb_params)
        time_model = train_lgb_time(X_train, y_time_train, lgb_params)
        winner_model = train_rf_classifier(X_train, y_win_train)

        X_val = X.iloc[val_idx]
        val_df = df.iloc[val_idx]

        position_preds = position_model.predict(X_val)
        time_preds = time_model.predict(X_val)
        winner_probs = winner_model.predict_proba(X_val)[:, 1]

        metrics = compute_fold_metrics(val_df, position_preds, time_preds, winner_probs)
        LOG.info(
            "Fold %s - Winner Acc: %.3f | Top3 Acc: %.3f | Top10 Acc: %.3f | Pos MAE: %.3f | Time MAE: %.3f | Spearman: %.3f",
            label,
            metrics["winner_accuracy"],
            metrics["top3_accuracy"],
            metrics["top10_accuracy"],
            metrics["position_mae"],
            metrics["time_mae"],
            metrics["spearman"],
        )


def main() -> None:
    setup_logging()
    args = parse_args()
    engine = get_engine(args.database_url)
    df = load_feature_matrix(engine)
    df = prepare_targets(df)
    features, _ = select_features(df)

    evaluate_models(
        df,
        features,
        df["finishing_position"],
        df["race_time_delta"],
        df["dnf"].astype(int),
        df["podium_class"].astype(int),
        df["winner"].astype(int),
        args.model_dir,
        args.n_splits,
        args.optuna_trials,
    )


if __name__ == "__main__":
    main()
