"""E1：站点内时间滞后特征（D-1/D-2/D-3、3/7 日滚动统计、3 日变化量）。

规则：
- 只在同一站点内部沿日历日回溯，绝不跨站点滚动；
- 先把每个站点重采样到完整日网格，再用 shift 取真实前一自然日，保证滞后是 D-1 而不是“上一条记录”；
- 所有滚动统计先 shift(1) 再滚动，窗口内不包含目标日；
- 缺测保持 NaN，不用未来值或插值补齐；
- 不使用 MDA8 或任何 O₃ 历史作为输入，避免目标观测依赖与标签泄漏。
"""

import pandas as pd

from . import config, models

POLLUTION_LAG = ["no2_mean", "co_mean", "pm25_mean", "pm10_mean", "so2_mean"]
METEO_LAG = ["t2m_mean", "rh_mean", "ws_mean", "u_mean", "v_mean", "srad_sum", "blh_mean"]
ROLL_VARS = POLLUTION_LAG + ["t2m_mean", "rh_mean", "ws_mean", "srad_sum", "blh_mean"]
CHANGE_VARS = ["t2m_mean", "srad_sum", "pm25_mean", "no2_mean"]


def add_lag_features(df: pd.DataFrame, max_lag: int = 3, windows: tuple[int, ...] = (3, 7)) -> pd.DataFrame:
    """返回在 df 上追加滞后特征的新表；仅使用目标日之前的历史。"""
    grid = pd.date_range(f"{config.STUDY_YEARS[0]}-01-01", f"{config.STUDY_YEARS[-1]}-12-31", freq="D")

    frames = []
    for code, g in df.groupby("station_code", sort=False):
        s = g.set_index("date").sort_index()
        s = s.reindex(grid)  # 缺失日期保持 NaN：滞后按自然日对齐，不追加上一条记录
        new = pd.DataFrame(index=s.index)

        for h in range(1, max_lag + 1):
            for col in POLLUTION_LAG + METEO_LAG:
                new[f"{col}_lag{h}"] = s[col].shift(h)

        base = s[ROLL_VARS].shift(1)  # 窗口只覆盖 D-1 及更早
        for w in windows:
            roll = base.rolling(w, min_periods=w)
            for col in ROLL_VARS:
                new[f"{col}_roll{w}_mean"] = roll[col].mean()
                new[f"{col}_roll{w}_std"] = roll[col].std()

        for col in CHANGE_VARS:
            new[f"{col}_chg3"] = s[col].shift(1) - s[col].shift(1 + max_lag)

        new["station_code"] = code
        new = new.reset_index().rename(columns={"index": "date"})
        frames.append(new)

    lags = pd.concat(frames, ignore_index=True)
    return df.merge(lags, on=["station_code", "date"], how="left", validate="one_to_one")


def feature_tiers(base: list[str] | None = None) -> dict[str, list[str]]:
    """E1-0…E1-3 四级特征清单（每级都在 F1 之上只增加时间历史信息）。"""
    base = base or models.FEATURE_GROUPS["F1"]
    d1 = [f"{c}_lag1" for c in POLLUTION_LAG + METEO_LAG]
    d23 = [f"{c}_lag{h}" for h in (2, 3) for c in POLLUTION_LAG + METEO_LAG]
    roll = [f"{c}_roll{w}_{stat}" for c in ROLL_VARS for w in (3, 7) for stat in ("mean", "std")]
    chg = [f"{c}_chg3" for c in CHANGE_VARS]
    return {
        "E1-0": list(base),
        "E1-1": list(base) + d1,
        "E1-2": list(base) + d1 + d23,
        "E1-3": list(base) + d1 + d23 + roll + chg,
    }
