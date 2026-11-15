"""步骤 12：E2 TROPOMI 卫星特征增量实验。

前提：已运行 11_merge_tropomi.py，得到 data/processed/modeling_table_with_tropomi.parquet。
核心原则：所有版本在**同一批“HCHO 与 NO₂ 同时有效”的站点日**上比较，避免样本减少造成的表观改善。
协议：2019—2020 训练 → 2021 选参 → 2019—2021 重训 → 2022 测试；嵌套留城市（外层城市不参与调参）。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fjozone import config, models  # noqa: E402

COASTAL = {"福州", "厦门", "泉州", "漳州", "莆田", "宁德"}
INLAND = {"南平", "三明", "龙岩"}
N_BOOT = 2000
SEED = 42


def _versions(df: pd.DataFrame) -> dict[str, list[str]]:
    base = models.FEATURE_GROUPS["F1"]
    extra_quality = [c for c in ("s5p_quality",) if c in df.columns]
    return {
        "E2-0": list(base),
        "E2-1": list(base) + ["hcho_col"],
        "E2-2": list(base) + ["no2_col"],
        "E2-3": list(base) + ["hcho_col", "no2_col"],
        "E2-4": list(base) + ["hcho_col", "no2_col", "hcho_no2_ratio"] + extra_quality,
    }


def _bootstrap(y, p, p_base) -> dict:
    rng = np.random.default_rng(SEED)
    n = len(y)
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = rng.integers(0, n, n)
        d[b] = np.sqrt(np.mean((y[s] - p[s]) ** 2)) - np.sqrt(np.mean((y[s] - p_base[s]) ** 2))
    return {"dRMSE": d.mean(), "ci_lo": np.percentile(d, 2.5), "ci_hi": np.percentile(d, 97.5),
            "p_lt0": float((d < 0).mean())}


def main() -> None:
    table = pd.read_parquet(config.DATA_PROCESSED / "modeling_table_with_tropomi.parquet")
    coverage = table[["hcho_col", "no2_col"]].notna().all(axis=1)
    df = table[coverage].copy()
    print(f"共同样本: {len(df):,} 站点日（覆盖率 {coverage.mean():.2%}，总 {len(table):,}）", flush=True)
    if len(df) < 0.2 * len(table):
        print("[警告] 卫星共同样本不足 20%，结论需谨慎，建议改用 L3 更宽缓冲或降低质量筛选", flush=True)
    if df["year"].nunique() < 4:
        print(f"[警告] 共同样本仅覆盖年份 {sorted(df['year'].unique())}", flush=True)

    tiers = _versions(df)
    print("E2 版本特征数:", {k: len(v) for k, v in tiers.items()}, flush=True)
    zoo = models.model_zoo()
    zoo_sp = {k: zoo[k] for k in ("B2_XGB", "B3_LGBM")}

    time_tables, city_tables, pooled_tables, preds = [], [], [], {}
    for ver, feats in tiers.items():
        print(f"\n=== {ver}（{len(feats)} 特征）时间协议 ===", flush=True)
        res = models.run_feature_set(df, feats, zoo)
        t = res["table"].copy()
        t.insert(0, "version", ver)
        time_tables.append(t)
        preds[ver] = res["preds"][("B2_XGB", "test")].copy()

        print(f"=== {ver} 嵌套留城市 ===", flush=True)
        per_city, pooled = models.run_spatial_nested(df, feats, zoo_sp)
        per_city.insert(0, "version", ver)
        pooled.insert(0, "version", ver)
        city_tables.append(per_city)
        pooled_tables.append(pooled)

    time_tab = pd.concat(time_tables, ignore_index=True)
    city_tab = pd.concat(city_tables, ignore_index=True)
    pooled_tab = pd.concat(pooled_tables, ignore_index=True)
    time_tab.to_csv(config.TABLES / "e2_time_validation.csv", index=False, encoding="utf-8-sig")
    city_tab.to_csv(config.TABLES / "e2_spatial_nested_by_city.csv", index=False, encoding="utf-8-sig")
    pooled_tab.to_csv(config.TABLES / "e2_spatial_nested_pooled.csv", index=False, encoding="utf-8-sig")

    # ---- 相对 E2-0 的增量 ----
    test = time_tab[time_tab["split"] == "test"]
    base = test[test["version"] == "E2-0"].set_index("model")
    rows = []
    for ver in tiers:
        cur = test[test["version"] == ver].set_index("model")
        for m in cur.index:
            rows.append({
                "version": ver, "model": m, "RMSE": cur.loc[m, "RMSE"], "R2": cur.loc[m, "R2"],
                "MAE": cur.loc[m, "MAE"], "RMSE_P95": cur.loc[m, "RMSE_P95"],
                "dRMSE": cur.loc[m, "RMSE"] - base.loc[m, "RMSE"],
                "dRMSE_pct": (cur.loc[m, "RMSE"] / base.loc[m, "RMSE"] - 1) * 100,
            })
    delta_time = pd.DataFrame(rows)

    sp_base = pooled_tab[pooled_tab["version"] == "E2-0"].set_index("model")
    rows = []
    for ver in tiers:
        cur = pooled_tab[pooled_tab["version"] == ver].set_index("model")
        for m in cur.index:
            rows.append({
                "version": ver, "model": m, "RMSE": cur.loc[m, "RMSE"], "R2": cur.loc[m, "R2"],
                "RMSE_P95": cur.loc[m, "RMSE_P95"],
                "dRMSE": cur.loc[m, "RMSE"] - sp_base.loc[m, "RMSE"],
                "dRMSE_pct": (cur.loc[m, "RMSE"] / sp_base.loc[m, "RMSE"] - 1) * 100,
            })
    delta_sp = pd.DataFrame(rows)

    # ---- 沿海 / 内陆分解（XGB）----
    city_xgb = city_tab[city_tab["model"] == "B2_XGB"].copy()
    city_xgb["region"] = np.where(city_xgb["city"].isin(COASTAL), "沿海", "内陆")
    city_xgb["weight_rmse2"] = city_xgb["n"] * city_xgb["RMSE"] ** 2
    region = city_xgb.groupby(["version", "region"]).apply(
        lambda g: pd.Series({"n": g["n"].sum(), "RMSE": np.sqrt(g["weight_rmse2"].sum() / g["n"].sum())}),
        include_groups=False).reset_index()

    # ---- 配对自助法（E2 各版本 vs E2-0）----
    y = preds["E2-0"]["y_true"].to_numpy()
    p_base = preds["E2-0"]["y_pred"].to_numpy()
    boot_rows = []
    for ver in list(tiers)[1:]:
        p = preds[ver].reindex(preds["E2-0"].index)
        boot = _bootstrap(p["y_true"].to_numpy(), p["y_pred"].to_numpy(), p_base)
        boot_rows.append({"version": ver, **boot})
    boot = pd.DataFrame(boot_rows)

    delta_time.to_csv(config.TABLES / "e2_delta_time.csv", index=False, encoding="utf-8-sig")
    delta_sp.to_csv(config.TABLES / "e2_delta_spatial.csv", index=False, encoding="utf-8-sig")
    region.to_csv(config.TABLES / "e2_region_spatial.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(config.TABLES / "e2_bootstrap.csv", index=False, encoding="utf-8-sig")

    xgb_t = delta_time[delta_time["model"] == "B2_XGB"].set_index("version")
    xgb_s = delta_sp[delta_sp["model"] == "B2_XGB"].set_index("version")
    keep_rows = []
    for ver in ["E2-1", "E2-2", "E2-3", "E2-4"]:
        keep_rows.append({
            "version": ver,
            "时间ΔRMSE": xgb_t.loc[ver, "dRMSE"], "时间Δ%": xgb_t.loc[ver, "dRMSE_pct"],
            "空间ΔRMSE": xgb_s.loc[ver, "dRMSE"], "空间Δ%": xgb_s.loc[ver, "dRMSE_pct"],
            "P95Δ": xgb_t.loc[ver, "RMSE_P95"] - xgb_t.loc["E2-0", "RMSE_P95"],
        })
    keep_tab = pd.DataFrame(keep_rows)
    keep_tab["通过(≥0.5μg且≥1%)"] = (keep_tab["时间ΔRMSE"] <= -0.5) & (keep_tab["空间ΔRMSE"] <= -0.5)

    lines = [
        "# E2 TROPOMI 卫星特征增量实验",
        "",
        "## 设计要点",
        "",
        f"- 共同样本：HCHO 与 NO₂ 同时有效的 {len(df):,} 个站点日（占建模表 {len(df) / len(table):.2%}），五个版本严格同一行集。",
        "- 卫星值保留原始 mol/m²，不插补、不置零、不截断负值；HCHO/NO₂ 仅在两者均为正时计算。",
        "- HCHO/NO₂ 只作为化学状态代理指标，不表述为 VOC/NOx 敏感性分类。",
        "- 时间协议：2019—2020 训练 → 2021 选参 → 2019—2021 重训 → 2022 测试。",
        "- 空间协议：嵌套留城市，被留城市不参与调参（`models.run_spatial_nested`）。",
        "",
        "## 2022 独立测试（相对 E2-0）",
        "",
        delta_time.round(3).to_markdown(index=False),
        "",
        "## 嵌套留城市（9 折合并，XGB/LGBM）",
        "",
        delta_sp.round(3).to_markdown(index=False),
        "",
        "## 沿海 / 内陆分解（XGB）",
        "",
        region.round(3).to_markdown(index=False),
        "",
        "## 配对自助法（2022，相对 E2-0）",
        "",
        boot.round(3).to_markdown(index=False),
        "",
        "## 保留判定（时间与空间需同时达到 0.5 μg/m³ 且 1%）",
        "",
        keep_tab.round(3).to_markdown(index=False),
        "",
        "## 判读原则",
        "",
        "- 时间改善而空间不改善：不作为核心创新，只记录为诊断型增益。",
        "- 空间改善而时间不改善：进一步检查沿海—内陆差异，判断是否来自区域化学背景而非站点信息。",
        "- 仅 HCHO 有效：保留 HCHO，放弃无效的 NO₂ 组合。",
        "- HCHO 与 NO₂ 均无稳定增益：不保留 E2，转向 E3/E4。",
        "",
        "## 注意",
        "",
        "- 卫星柱浓度与地面 MDA8 的时间匹配为同日，属于诊断型估算，不构成提前预报。",
        "- TROPOMI L3 的 1.1 km 网格不等于独立观测分辨率（原始足迹约 3.5×7 km）。",
        "",
    ]
    (config.REPORTS / "e2_tropomi_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n报告: outputs/reports/e2_tropomi_report.md")
    print(keep_tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
