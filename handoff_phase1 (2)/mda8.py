"""MDA8（日最大 8 小时滑动平均）计算。

规则（北京时间 UTC+8，日均按自然日 00:00-24:00）：
- 8 小时滑动窗口起点为 0..16 时，共 17 个窗口；
- 每个窗口需至少 6 个有效小时（HJ 663-2013 的 75% 规则）；
- 日内 MDA8 = 有效窗口中的最大值；
- 日内有效窗口数 >= 6 才视为当日 MDA8 有效（严格阈值 >=13 同时记录）。
"""

import warnings

import numpy as np
import pandas as pd

from . import config


def compute_mda8(hourly: pd.DataFrame) -> pd.DataFrame:
    df = hourly.copy()
    df["date"] = df["dt"].dt.normalize()
    df["h"] = df["dt"].dt.hour.astype(int)

    # 同一站点同一小时重复观测时取均值
    wide = (
        df.pivot_table(index=["station_code", "date"], columns="h", values="o3", aggfunc="mean", dropna=False)
        .reindex(columns=range(24))
        .astype(float)
    )
    vals = wide.to_numpy()
    valid_hours = np.isfinite(vals).sum(axis=1)

    n_rows = vals.shape[0]
    win_means = np.full((n_rows, 17), np.nan)
    win_counts = np.zeros((n_rows, 17), dtype=int)

    filled = np.where(np.isfinite(vals), vals, 0.0)
    mask = np.isfinite(vals).astype(int)

    for s in range(17):
        seg_sum = filled[:, s : s + 8].sum(axis=1)
        seg_cnt = mask[:, s : s + 8].sum(axis=1)
        win_counts[:, s] = seg_cnt
        ok = seg_cnt >= config.MIN_HOURS_PER_WINDOW
        win_means[ok, s] = seg_sum[ok] / seg_cnt[ok]

    n_valid_windows = np.isfinite(win_means).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mda8 = np.nanmax(win_means, axis=1)

    safe = np.where(np.isfinite(win_means), win_means, -np.inf)
    best_start = np.argmax(safe, axis=1)
    best_start = np.where(n_valid_windows > 0, best_start, -1)

    out = pd.DataFrame(
        {
            "station_code": wide.index.get_level_values(0),
            "date": wide.index.get_level_values(1),
            "mda8": mda8,
            "mda8_raw": mda8,
            "n_valid_hours": valid_hours,
            "n_valid_windows": n_valid_windows,
            "mda8_window_start_hour": best_start,
        }
    )
    out["day_valid"] = out["n_valid_windows"] >= config.MIN_WINDOWS_PER_DAY
    out["day_valid_strict"] = out["n_valid_windows"] >= config.MIN_WINDOWS_PER_DAY_STRICT
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean24 = np.nanmean(vals, axis=1)
    out["o3_mean24"] = pd.Series(np.where(valid_hours > 0, mean24, np.nan))
    out.loc[~out["day_valid"], "mda8"] = np.nan
    out["year"] = out["date"].dt.year
    return out.reset_index(drop=True)
