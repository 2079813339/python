"""读取 quotsoft 逐日全国监测点 CSV，提取福建站点常规污染物日均值。

逐日文件为宽表：date, hour, type, 站点编码..., 每行一个“时刻-污染物类型”。
O3 的官方 8 小时滑动均值列（O3_8h）与日最大 8 小时值（O3_8h_24h）也一并记录，
用于与自算 MDA8 交叉校验。
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

DAILY_TYPES = ["NO2", "CO", "PM2.5", "PM10", "SO2", "O3"]
MIN_HOURS_FOR_DAILY = 18


def _read_day(path: Path, codes: list[str]) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype={"date": "Int64", "hour": "Int64"})
    except Exception:  # noqa: BLE001
        try:
            df = pd.read_csv(path, encoding="gbk", dtype={"date": "Int64", "hour": "Int64"})
        except Exception:  # noqa: BLE001
            return None
    df.columns = [str(c).strip() for c in df.columns]
    if "date" not in df.columns or "type" not in df.columns:
        return None
    present = [c for c in codes if c in df.columns]
    if not present:
        return None

    rows: dict[str, pd.Series] = {}
    types = df["type"].astype(str).str.upper().str.strip()
    # 实时发布数据中 00 时的 O3_8h / O3_8h_24h 仍是前一日的值，计算日均/日最大时需剔除
    keep_row = pd.Series(True, index=df.index)
    if "hour" in df.columns:
        keep_row = df["hour"].fillna(-1).astype(int) != 0
    for t in DAILY_TYPES + ["O3_8H_24H", "O3_8H"]:
        mask = types == t
        if t in ("O3_8H_24H", "O3_8H"):
            mask = mask & keep_row
        sub = df.loc[mask, present]
        if sub.empty:
            continue
        vals = sub.apply(pd.to_numeric, errors="coerce")
        if t == "O3_8H_24H":
            # CNEMC 实时发布中的 O3_8h_24h 为逐时滚动最大值（截至当前小时的日最大 8 小时均值），
            # 日内取最大值即为该日官方 MDA8
            rows["o3_mda8_official"] = vals.max(skipna=True)
            continue
        if t == "O3_8H":
            rows["o3_8h_max"] = vals.max(skipna=True)
            continue
        mean = vals.mean(skipna=True)
        cnt = vals.count()
        rows[f"{t.lower().replace('.', '')}_mean"] = mean
        rows[f"{t.lower().replace('.', '')}_nhour"] = cnt

    date = int(df["date"].dropna().iloc[0]) if df["date"].notna().any() else None
    if date is None or not rows:
        return None
    out = pd.DataFrame(rows)
    out.insert(0, "date", pd.to_datetime(str(date), format="%Y%m%d"))
    out.insert(0, "station_code", out.index.astype(str))
    return out.reset_index(drop=True)


def load_from_wide(path: Path, codes: list[str], years: list[int]) -> pd.DataFrame:
    """从多污染物 ZIP 盘点输出的逐时宽表计算日均值（每类污染物要求当日 ≥18 个有效小时）。"""
    cols = ["dt", "station_id", "O3", "NO2", "SO2", "CO", "PM25", "PM10"]
    df = pd.read_csv(path, parse_dates=["dt"], usecols=[c for c in cols])
    df = df[df["station_id"].astype(str).isin(codes) & df["dt"].dt.year.isin(years)].copy()
    df = df.rename(columns={"station_id": "station_code"})
    df["date"] = df["dt"].dt.normalize()
    spec = {"o3": "O3", "no2": "NO2", "co": "CO", "pm25": "PM25", "pm10": "PM10", "so2": "SO2"}
    agg = {}
    for key, col in spec.items():
        agg[f"{key}_mean"] = (col, "mean")
        agg[f"{key}_nhour"] = (col, "count")
    df = df.groupby(["station_code", "date"], as_index=False).agg(**agg)
    for key in spec:
        df.loc[df[f"{key}_nhour"] < MIN_HOURS_FOR_DAILY, f"{key}_mean"] = np.nan
    return df.sort_values(["station_code", "date"]).reset_index(drop=True)


def load_pollutants(codes: list[str], years: list[int], directory: Path | None = None) -> pd.DataFrame:
    directory = directory or config.QUOTSOFT_DIR
    files = sorted(directory.glob("china_sites_*.csv"))
    files = [f for f in files if int(f.stem.split("_")[-1][:4]) in years]
    if not files:
        raise FileNotFoundError(f"{directory} 下没有目标年份的 quotsoft 逐日文件")

    frames = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for f in files:
            part = _read_day(f, codes)
            if part is not None:
                frames.append(part)
    if not frames:
        raise RuntimeError("未能从 quotsoft 文件解析出福建站点数据")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(["station_code", "date"]).sort_values(["station_code", "date"])

    # 日均值有效小时数不足时置缺测，并保留计数列
    for t in DAILY_TYPES:
        key = f"{t.lower().replace('.', '')}_mean"
        cnt = f"{t.lower().replace('.', '')}_nhour"
        if key in df.columns and cnt in df.columns:
            df.loc[df[cnt] < MIN_HOURS_FOR_DAILY, key] = np.nan
    return df.reset_index(drop=True)
