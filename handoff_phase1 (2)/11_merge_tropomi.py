"""步骤 11：把 GEE 导出的 TROPOMI 站点日表并入建模表（字段自适应）。

用法：
    python scripts/11_merge_tropomi.py --input outputs/e2/fujian_tropomi_station_daily_2019_2022.csv

脚本会自动识别站点编码列、日期列、HCHO/NO₂ 列与缓冲区半径（3/5/10 km），并把结果归一化为
hcho_col / no2_col / hcho_no2_ratio / s5p_quality_x 等固定列名，便于 E2 实验直接引用。
输出：
    data/processed/tropomi_station_daily.parquet   归一化卫星表
    outputs/e2/tropomi_merge_audit.txt             覆盖率与质量审计
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fjozone import config  # noqa: E402

STATION_KEYS = ("station_code", "stationid", "station_id", "code", "siteid", "site_id")
DATE_KEYS = ("date", "day", "system:time_start", "time_start", "datetime")


def _canon(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(name).lower())


def _find(df: pd.DataFrame, keys: tuple[str, ...], contains: tuple[tuple[str, ...], ...] = ()) -> str | None:
    canon = {_canon(c): c for c in df.columns}
    for k in keys:
        if k in canon:
            return canon[k]
    for parts in contains:
        for c, orig in canon.items():
            if all(p in c for p in parts):
                return orig
    return None


def _radius(col: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*km", str(col), flags=re.I)
    return float(m.group(1)) if m else None


def _pick(group: list[str], prefer_km: float) -> tuple[str | None, list[str]]:
    """在候选列中优先选指定缓冲区半径；无半径信息时取第一列。"""
    if not group:
        return None, []
    exact = [c for c in group if _radius(c) == prefer_km]
    if exact:
        return exact[0], group
    if all(_radius(c) is None for c in group):
        return group[0], group
    return None, group


def merge(path: Path, prefer_km: float = 5.0) -> None:
    raw = pd.read_csv(path)
    print(f"[读取] {path.name}: {raw.shape[0]:,} 行 × {raw.shape[1]} 列")

    station_col = _find(raw, STATION_KEYS, (("station", "code"), ("station", "id"), ("site", "id")))
    date_col = _find(raw, DATE_KEYS, (("date",),))
    if station_col is None or date_col is None:
        raise SystemExit(f"无法识别站点列或日期列，实际列名: {list(raw.columns)[:20]}")

    df = pd.DataFrame({
        "station_code": raw[station_col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True),
        "date": pd.to_datetime(raw[date_col].astype(str), errors="coerce").dt.normalize(),
    })
    if df["date"].isna().any():
        # 兼容 YYYYMMDD 这类缺失零填充的写法
        alt = pd.to_datetime(raw[date_col].astype(str).str.replace("-", ""), format="%Y%m%d", errors="coerce")
        df["date"] = df["date"].fillna(alt.dt.normalize())
    if df["date"].isna().any():
        raise SystemExit(f"日期解析失败 {int(df['date'].isna().sum())} 行，示例: {raw[date_col].head(3).tolist()}")

    hcho_all = [c for c in raw.columns if "hcho" in _canon(c) and "ratio" not in _canon(c)]
    no2_all = [c for c in raw.columns if "no2" in _canon(c) and "ratio" not in _canon(c)]
    hcho_col, _ = _pick(hcho_all, prefer_km)
    no2_col, _ = _pick(no2_all, prefer_km)
    if hcho_col is None or no2_col is None:
        raise SystemExit(f"未能在 {prefer_km} km 缓冲区识别 HCHO/NO₂ 列；候选 HCHO={hcho_all}，NO₂={no2_all}")

    canon = {_canon(c): c for c in raw.columns}
    qual_col = next((orig for c, orig in canon.items() if any(p in c for p in ("quality", "validpixel", "qa", "fraction"))), None)

    df["hcho_col"] = pd.to_numeric(raw[hcho_col], errors="coerce")
    df["no2_col"] = pd.to_numeric(raw[no2_col], errors="coerce")
    df["hcho_col_km"] = _radius(hcho_col) or np.nan
    df["no2_col_km"] = _radius(no2_col) or np.nan
    both_pos = (df["hcho_col"] > 0) & (df["no2_col"] > 0)
    df["hcho_no2_ratio"] = np.where(both_pos, df["hcho_col"] / df["no2_col"], np.nan)
    if qual_col is not None:
        df["s5p_quality"] = pd.to_numeric(raw[qual_col], errors="coerce")

    df = df.dropna(subset=["date"]).sort_values(["station_code", "date"]).reset_index(drop=True)
    dup = int(df.duplicated(["station_code", "date"]).sum())
    df = df.drop_duplicates(["station_code", "date"], keep="first")

    model = pd.read_parquet(config.DATA_PROCESSED / "modeling_table.parquet")
    keys = model[["station_code", "date", "city", "year", "mda8"]]
    merged = keys.merge(df, on=["station_code", "date"], how="left")

    n_model = len(model)
    both_valid = merged["hcho_col"].notna() & merged["no2_col"].notna()
    lines = [
        "TROPOMI 站点日数据合并审计",
        "=" * 60,
        f"输入文件: {path.name}",
        f"原始记录: {raw.shape[0]:,} 行 × {raw.shape[1]} 列；重复站点日 {dup} 条（保留首条）",
        f"识别列: 站点='{station_col}', 日期='{date_col}', HCHO='{hcho_col}', NO₂='{no2_col}', 质量列={qual_col or '无'}",
        f"缓冲区半径: HCHO={df['hcho_col_km'].iloc[0]} km, NO₂={df['no2_col_km'].iloc[0]} km（偏好 {prefer_km} km）",
        "",
        f"卫星表站点数: {df['station_code'].nunique()}，日期范围 {df['date'].min().date()} ~ {df['date'].max().date()}",
        f"建模表站点日: {n_model:,}；两变量同时有效: {int(both_valid.sum()):,}（覆盖率 {both_valid.mean():.2%}）",
        f"HCHO 有效率: {merged['hcho_col'].notna().mean():.2%}；NO₂ 有效率: {merged['no2_col'].notna().mean():.2%}",
        f"HCHO 负值比例: {(merged['hcho_col'] < 0).mean():.2%}；NO₂ 负值比例: {(merged['no2_col'] < 0).mean():.2%}",
        f"比值为正且可计算: {int(merged['hcho_no2_ratio'].notna().sum()):,}（{merged['hcho_no2_ratio'].notna().mean():.2%}）",
        "",
        "逐年覆盖:",
        merged.groupby("year").apply(
            lambda g: pd.Series({
                "站点日": len(g),
                "HCHO有效": g["hcho_col"].notna().sum(),
                "HCHO覆盖率": f"{g['hcho_col'].notna().mean():.1%}",
                "NO2有效": g["no2_col"].notna().sum(),
                "NO2覆盖率": f"{g['no2_col'].notna().mean():.1%}",
            }), include_groups=False).to_string(),
        "",
        "逐城市覆盖（HCHO有效比例）:",
        merged.groupby("city").apply(
            lambda g: pd.Series({"站点日": len(g), "HCHO有效比例": f"{g['hcho_col'].notna().mean():.1%}"}),
            include_groups=False).to_string(),
        "",
        "柱浓度分布（μg/m³ 不适用，单位 mol/m²）:",
        merged[["hcho_col", "no2_col", "hcho_no2_ratio"]].describe().round(6).to_string(),
        "",
        "说明: 未做插补、未把缺测或负值置零、未截断极值；HCHO/NO₂ 仅作化学状态代理，不构成敏感性分类。",
    ]
    text = "\n".join(str(x) for x in lines)
    (config.OUTPUTS / "e2").mkdir(parents=True, exist_ok=True)
    (config.OUTPUTS / "e2" / "tropomi_merge_audit.txt").write_text(text, encoding="utf-8")
    df.to_parquet(config.DATA_PROCESSED / "tropomi_station_daily.parquet", index=False)
    merged.to_parquet(config.DATA_PROCESSED / "modeling_table_with_tropomi.parquet", index=False)
    print(text)
    print("\n输出: data/processed/modeling_table_with_tropomi.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/e2/fujian_tropomi_station_daily_2019_2022.csv")
    ap.add_argument("--km", type=float, default=5.0)
    args = ap.parse_args()
    merge(Path(args.input), args.km)
