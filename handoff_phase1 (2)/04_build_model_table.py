"""步骤 4：合并 MDA8、气象、常规污染物，生成统一建模数据表并报告特征组可用性。"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fjozone import config, models, pollutants, stations  # noqa: E402


def main() -> None:
    fj = stations.fujian_stations()
    codes = fj["station_code"].tolist()
    met = pd.read_parquet(config.DATA_INTERIM / "fj_meteo_daily.parquet")

    audit_dir = config.OUTPUTS / "fujian_audit"
    audit_daily = audit_dir / "fujian_daily_mda8.csv"
    audit_hourly = audit_dir / "fujian_hourly_wide.csv"
    if audit_daily.exists() and audit_hourly.exists():
        # 主数据源：CNEMC 多污染物年度 ZIP（经 06_audit_cnemc_zips.py 盘点）
        daily = pd.read_csv(audit_daily, parse_dates=["date"]).rename(
            columns={"station_id": "station_code", "mda8_ugm3": "mda8", "mda8_day_valid": "day_valid",
                     "longitude": "lon", "latitude": "lat"}
        )
        daily.loc[~daily["day_valid"].astype(bool), "mda8"] = float("nan")
        daily = daily.merge(fj[["station_code", "elevation"]], on="station_code", how="left")
        pol = pollutants.load_from_wide(audit_hourly, codes, config.STUDY_YEARS)
        pol.to_parquet(config.DATA_INTERIM / "fj_pollutants_daily.parquet", index=False)
        print(f"[主数据源] 多污染物 ZIP 盘点表：标签 {len(daily):,} 站点日，污染物日均 {pol.shape}")
    else:
        daily = pd.read_parquet(config.DATA_INTERIM / "fj_mda8_2019_2022.parquet")
        pol = None
        files = list(config.QUOTSOFT_DIR.glob("china_sites_*.csv"))
        if len(files) >= 300:
            pol = pollutants.load_pollutants(codes, config.STUDY_YEARS)
            pol.to_parquet(config.DATA_INTERIM / "fj_pollutants_daily.parquet", index=False)
            print(f"常规污染物日均表: {pol.shape}")
        else:
            print(f"[警告] quotsoft 逐日文件仅 {len(files)} 个，F1 特征组暂不可用")

    df = daily.merge(met, on=["station_code", "date"], how="left", suffixes=("", "_met"))
    if pol is not None:
        # 仅合并污染物日均值作为特征；O3 相关列保留在 interim 表中用于一致性核查，避免标签泄漏
        keep = ["station_code", "date"] + [c for c in ["no2_mean", "co_mean", "pm25_mean", "pm10_mean", "so2_mean"] if c in pol.columns]
        df = df.merge(pol[keep], on=["station_code", "date"], how="left")

    df = models.add_calendar(df)
    df = df[df["mda8"].notna()].copy()
    # 气象特征全部缺失的样本不参与建模
    core_meteo = ["t2m_mean", "rh_mean", "sp_mean", "precip_sum"]
    df = df.dropna(subset=core_meteo).reset_index(drop=True)

    df.to_parquet(config.DATA_PROCESSED / "modeling_table.parquet", index=False)
    df.to_csv(config.DATA_PROCESSED / "modeling_table.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")

    print(f"\n建模样本: {len(df):,} 站点日，站点 {df['station_code'].nunique()}，城市 {df['city'].nunique()}")
    print(df.groupby("year").agg(n=("mda8", "size"), mda8_mean=("mda8", "mean"), mda8_max=("mda8", "max")).round(2).to_string())

    print("\n特征组可用性：")
    for name, feats in models.FEATURE_GROUPS.items():
        missing = [f for f in feats if f not in df.columns]
        null_rate = df[[f for f in feats if f in df.columns]].isna().mean().mean() if feats else 1.0
        status = "可用" if not missing else f"缺列 {missing}"
        print(f"  {name}: {len(feats)} 个特征，平均缺失率 {null_rate:.2%}，{status}")


if __name__ == "__main__":
    main()
