"""基准模型（B0-B4）、特征组（F0-F2）、时间与空间验证协议、分层评价指标。"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from . import config

METEO_FEATURES = [
    "t2m_mean", "t2m_max", "t2m_min", "dew_mean", "rh_mean", "sp_mean",
    "precip_sum", "cloud_mean", "srad_sum", "ws_mean", "u_mean", "v_mean",
    "blh_max", "blh_mean", "vpd_max", "soil_moist_mean",
]
CALENDAR_FEATURES = ["month", "doy", "doy_sin", "doy_cos"]
GEO_FEATURES = ["lon", "lat", "elevation"]
POLLUTANT_FEATURES = ["no2_mean", "co_mean", "pm25_mean", "pm10_mean", "so2_mean"]
SATELLITE_FEATURES: list[str] = []  # F2：TROPOMI HCHO/NO2 等，当前数据未接入

FEATURE_GROUPS = {
    "F0": GEO_FEATURES + CALENDAR_FEATURES + METEO_FEATURES,
    "F1": GEO_FEATURES + CALENDAR_FEATURES + METEO_FEATURES + POLLUTANT_FEATURES,
}
if SATELLITE_FEATURES:  # F2 仅在卫星特征实际接入后才参与比较
    FEATURE_GROUPS["F2"] = FEATURE_GROUPS["F1"] + SATELLITE_FEATURES


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["doy"] = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    return df


def metrics(y_true, y_pred, prefix: str = "") -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    out = {
        f"{prefix}n": int(len(y_true)),
        f"{prefix}R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        f"{prefix}RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))) if len(y_true) else np.nan,
        f"{prefix}MAE": float(mean_absolute_error(y_true, y_pred)) if len(y_true) else np.nan,
    }
    for q in (90, 95):
        thr = np.nanpercentile(y_true, q) if len(y_true) else np.nan
        sel = y_true >= thr
        out[f"{prefix}n_P{q}"] = int(sel.sum())
        out[f"{prefix}RMSE_P{q}"] = float(np.sqrt(mean_squared_error(y_true[sel], y_pred[sel]))) if sel.sum() else np.nan
        out[f"{prefix}MAE_P{q}"] = float(mean_absolute_error(y_true[sel], y_pred[sel])) if sel.sum() else np.nan
    return out


def _sklearn_pipeline(est) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", est)])


def model_zoo(seed: int = 42) -> dict[str, tuple]:
    """返回 {模型名: (构造函数, 超参数候选列表, 是否需要缺失值填补)}。"""
    return {
        "B0_Linear": (lambda p: _sklearn_pipeline(LinearRegression()), [{}], True),
        "B1_RF": (
            lambda p: _sklearn_pipeline(RandomForestRegressor(random_state=seed, n_jobs=-1, **p)),
            [{"n_estimators": 300, "min_samples_leaf": 1}, {"n_estimators": 300, "min_samples_leaf": 3}],
            True,
        ),
        "B2_XGB": (
            lambda p: XGBRegressor(
                random_state=seed, n_jobs=8, tree_method="hist", subsample=0.8, colsample_bytree=0.8,
                learning_rate=0.05, n_estimators=400, **p,
            ),
            [{"max_depth": 6}, {"max_depth": 8}],
            False,
        ),
        "B3_LGBM": (
            lambda p: LGBMRegressor(random_state=seed, n_jobs=8, learning_rate=0.05, n_estimators=500,
                                    min_child_samples=20, verbose=-1, **p),
            [{"num_leaves": 31}, {"num_leaves": 63}],
            False,
        ),
        "B4_ET": (
            lambda p: _sklearn_pipeline(ExtraTreesRegressor(random_state=seed, n_jobs=-1, **p)),
            [{"n_estimators": 300, "min_samples_leaf": 1}, {"n_estimators": 300, "min_samples_leaf": 3}],
            True,
        ),
    }


def select_hyperparams(train: pd.DataFrame, val: pd.DataFrame, features: list[str], zoo: dict) -> dict:
    """在 2021 验证年上按 RMSE 选择超参数；不接触 2022 测试年。"""
    chosen = {}
    x_tr, y_tr = train[features], train["mda8"]
    x_va, y_va = val[features], val["mda8"]
    for name, (factory, grid, _needs_imp) in zoo.items():
        best = None
        for params in grid:
            model = factory(params)
            model.fit(x_tr, y_tr)
            pred = model.predict(x_va)
            rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
            if best is None or rmse < best[0]:
                best = (rmse, params)
        chosen[name] = best[1]
        print(f"    {name}: val_RMSE={best[0]:.2f} params={best[1]}", flush=True)
    return chosen


def run_feature_set(df: pd.DataFrame, features: list[str], zoo: dict) -> dict:
    """时间协议：2019-2020 训练 / 2021 验证选参 / 2019-2021 重训 / 2022 独立测试。"""
    years = df["year"]
    train = df[years.isin(config.TRAIN_YEARS) & df["mda8"].notna()]
    val = df[(years == config.VAL_YEAR) & df["mda8"].notna()]
    test = df[(years == config.TEST_YEAR) & df["mda8"].notna()]
    fit = df[(years.isin(config.TRAIN_YEARS + [config.VAL_YEAR])) & df["mda8"].notna()]

    params = select_hyperparams(train, val, features, zoo)
    rows, preds, importances = [], {}, {}
    for name, (factory, _grid, _imp) in zoo.items():
        # 验证集指标必须来自「仅用 2019—2020 训练」的模型，避免 2021 进入训练造成的样本内假象
        pre_model = factory(params[name])
        pre_model.fit(train[features], train["mda8"])
        p_val = pre_model.predict(val[features])
        rows.append({"model": name, "split": "val", **metrics(val["mda8"], p_val)})
        preds[(name, "val")] = pd.DataFrame({
            "station_code": val["station_code"].values,
            "city": val["city"].values,
            "date": val["date"].values,
            "y_true": val["mda8"].values,
            "y_pred": np.asarray(p_val, dtype=float),
        })
        # 最终模型：2019—2021 重训，用于 2022 独立测试
        model = factory(params[name])
        model.fit(fit[features], fit["mda8"])
        for split_name, part in (("train", train), ("test", test)):
            p = model.predict(part[features])
            rows.append({"model": name, "split": split_name, **metrics(part["mda8"], p)})
            preds[(name, split_name)] = pd.DataFrame({
                "station_code": part["station_code"].values,
                "city": part["city"].values,
                "date": part["date"].values,
                "y_true": part["mda8"].values,
                "y_pred": np.asarray(p, dtype=float),
            })
        importances[name] = _importance(model, features)
    return {"table": pd.DataFrame(rows), "preds": preds, "importance": importances, "params": params}


def run_spatial(df: pd.DataFrame, features: list[str], zoo: dict, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """留城市验证：整座城市的全部年份都不进入训练集。"""
    rows, pred_frames = [], []
    for city in sorted(df["city"].unique()):
        hold = df["city"] == city
        train = df[~hold & df["mda8"].notna()]
        test = df[hold & df["mda8"].notna()]
        if test.empty or train.empty:
            continue
        for name, (factory, _grid, _imp) in zoo.items():
            model = factory(params[name])
            model.fit(train[features], train["mda8"])
            p = model.predict(test[features])
            rows.append({"model": name, "city": city, **metrics(test["mda8"], p)})
            pred_frames.append(pd.DataFrame({
                "model": name, "city": city,
                "station_code": test["station_code"].values, "date": test["date"].values,
                "y_true": test["mda8"].values, "y_pred": np.asarray(p, dtype=float),
            }))
    preds = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    pooled = []
    if not preds.empty:
        for name, grp in preds.groupby("model"):
            pooled.append({"model": name, **metrics(grp["y_true"], grp["y_pred"])})
    return pd.DataFrame(rows), pd.DataFrame(pooled)


def _importance(model, features: list[str]) -> pd.Series:
    """返回归一化后的特征重要性（各模型内部占比之和为 1，便于跨模型比较）。"""
    m = model
    if hasattr(m, "named_steps"):
        m = m.named_steps["model"]
    if hasattr(m, "feature_importances_"):
        s = pd.Series(m.feature_importances_, index=features, dtype=float)
    elif hasattr(m, "coef_"):
        s = pd.Series(np.abs(m.coef_), index=features, dtype=float)
    else:
        return pd.Series(dtype=float)
    return (s / s.sum()).sort_values(ascending=False) if s.sum() else s
