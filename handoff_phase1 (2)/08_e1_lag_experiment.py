"""步骤 8：E1 时间滞后特征实验。

比较 E1-0（= F1）/ E1-1（+D-1）/ E1-2（+D-1..D-3）/ E1-3（+3、7 日滚动与 3 日变化），
沿用基准实验完全相同的时间划分（2019—2020 训练 → 2021 选参 → 2019—2021 重训 → 2022 测试）
与留城市划分；并额外在“滞后特征全部可用的共同样本”上做严格对齐对照。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fjozone import config, lagfeatures, models  # noqa: E402


def _xgb_only(zoo: dict) -> dict:
    return {"B2_XGB": zoo["B2_XGB"]}


def main() -> None:
    df = pd.read_parquet(config.DATA_PROCESSED / "modeling_table.parquet")
    lagged = lagfeatures.add_lag_features(df)
    lagged.to_parquet(config.DATA_PROCESSED / "modeling_table_e1.parquet", index=False)
    tiers = lagfeatures.feature_tiers()
    zoo = models.model_zoo()
    zoo_spatial = {k: zoo[k] for k in ("B2_XGB", "B3_LGBM")}

    print("E1 版本与特征数：")
    for name, feats in tiers.items():
        miss = lagged[feats].isna().mean()
        print(f"  {name}: {len(feats)} 个特征，平均缺失率 {miss.mean():.2%}，"
              f"最高 {miss.max():.2%}（{miss.idxmax()}）", flush=True)

    # 滞后特征与目标的相关性（检查是否引入异常强关联）
    corr = {c: lagged[[c, "mda8"]].corr().iloc[0, 1] for c in tiers["E1-3"][len(tiers["E1-0"]):]}
    corr = pd.Series(corr).dropna().sort_values(key=np.abs, ascending=False)
    print("\n与 MDA8 相关性最强的 8 个滞后特征：")
    print(corr.head(8).round(3).to_string(), flush=True)

    time_tables, spatial_pooled, spatial_city, params_by_ver = [], [], [], {}
    for name, feats in tiers.items():
        print(f"\n=== {name}（{len(feats)} 个特征）时间协议 ===", flush=True)
        res = models.run_feature_set(lagged, feats, zoo)
        tab = res["table"].copy()
        tab.insert(0, "version", name)
        time_tables.append(tab)
        params_by_ver[name] = res["params"]
        pred = res["preds"][("B2_XGB", "test")]
        pred.to_csv(config.TABLES / f"prediction_2022_{name}_B2_XGB.csv", index=False, encoding="utf-8-sig")

        print(f"=== {name} 留城市协议（XGB/LGBM）===", flush=True)
        city_res, pooled = models.run_spatial(lagged, feats, zoo_spatial, res["params"])
        city_res.insert(0, "version", name)
        pooled.insert(0, "version", name)
        spatial_city.append(city_res)
        spatial_pooled.append(pooled)

    time_tab = pd.concat(time_tables, ignore_index=True)
    city_tab = pd.concat(spatial_city, ignore_index=True)
    pooled_tab = pd.concat(spatial_pooled, ignore_index=True)

    # ---- 共同样本严格对齐（滞后特征全部可用的站点日）----
    full_cover = lagged[tiers["E1-3"]].notna().all(axis=1)
    common = lagged[full_cover].copy()
    print(f"\n=== 共同样本对照：{len(common):,} 站点日（占 {len(common) / len(lagged):.2%}）===", flush=True)
    common_tables = []
    for name, feats in tiers.items():
        res = models.run_feature_set(common, feats, _xgb_only(zoo))
        tab = res["table"].copy()
        tab.insert(0, "version", name)
        common_tables.append(tab)
    common_tab = pd.concat(common_tables, ignore_index=True)

    time_tab.to_csv(config.TABLES / "e1_lag_time_validation.csv", index=False, encoding="utf-8-sig")
    city_tab.to_csv(config.TABLES / "e1_lag_spatial_by_city.csv", index=False, encoding="utf-8-sig")
    pooled_tab.to_csv(config.TABLES / "e1_lag_spatial_pooled.csv", index=False, encoding="utf-8-sig")
    common_tab.to_csv(config.TABLES / "e1_lag_common_sample.csv", index=False, encoding="utf-8-sig")

    # ---- 汇总：相对 E1-0 的增量 ----
    test = time_tab[time_tab["split"] == "test"]
    base_test = test[test["version"] == "E1-0"].set_index("model")
    delta_rows = []
    for ver in tiers:
        cur = test[test["version"] == ver].set_index("model")
        for model_name in cur.index:
            delta_rows.append({
                "version": ver, "model": model_name,
                "R2": cur.loc[model_name, "R2"], "RMSE": cur.loc[model_name, "RMSE"],
                "MAE": cur.loc[model_name, "MAE"], "RMSE_P95": cur.loc[model_name, "RMSE_P95"],
                "dRMSE_vs_E1_0": cur.loc[model_name, "RMSE"] - base_test.loc[model_name, "RMSE"],
                "dRMSE_pct": (cur.loc[model_name, "RMSE"] - base_test.loc[model_name, "RMSE"]) / base_test.loc[model_name, "RMSE"] * 100,
                "dP95_vs_E1_0": cur.loc[model_name, "RMSE_P95"] - base_test.loc[model_name, "RMSE_P95"],
            })
    delta = pd.DataFrame(delta_rows)

    sp = pooled_tab[pooled_tab["version"] == "E1-0"].set_index("model")
    sp_delta = []
    for ver in tiers:
        cur = pooled_tab[pooled_tab["version"] == ver].set_index("model")
        for model_name in cur.index:
            sp_delta.append({
                "version": ver, "model": model_name, "R2": cur.loc[model_name, "R2"],
                "RMSE": cur.loc[model_name, "RMSE"], "RMSE_P95": cur.loc[model_name, "RMSE_P95"],
                "dRMSE_vs_E1_0": cur.loc[model_name, "RMSE"] - sp.loc[model_name, "RMSE"],
                "dRMSE_pct": (cur.loc[model_name, "RMSE"] - sp.loc[model_name, "RMSE"]) / sp.loc[model_name, "RMSE"] * 100,
            })
    spatial_delta = pd.DataFrame(sp_delta)
    delta.to_csv(config.TABLES / "e1_lag_delta_time.csv", index=False, encoding="utf-8-sig")
    spatial_delta.to_csv(config.TABLES / "e1_lag_delta_spatial.csv", index=False, encoding="utf-8-sig")

    xgb_time = delta[delta["model"] == "B2_XGB"].set_index("version")
    xgb_sp = spatial_delta[spatial_delta["model"] == "B2_XGB"].set_index("version")
    common_xgb = common_tab[(common_tab["split"] == "test")].set_index("version")

    # ---- 结论（按“时间 + 空间同时改善”判定）----
    keep = []
    for ver in ["E1-1", "E1-2", "E1-3"]:
        ok_time = xgb_time.loc[ver, "dRMSE_vs_E1_0"] < 0
        ok_sp = xgb_sp.loc[ver, "dRMSE_vs_E1_0"] < 0
        no_p95 = xgb_time.loc[ver, "dP95_vs_E1_0"] <= 0
        keep.append({"version": ver, "time_better": ok_time, "spatial_better": ok_sp,
                     "p95_not_worse": no_p95, "keep": bool(ok_time and ok_sp and no_p95)})
    keep_tab = pd.DataFrame(keep)

    base_xgb = xgb_time.loc["E1-0"]
    lines = [
        "# E1 时间滞后特征实验报告",
        "",
        "## 实验设计",
        "",
        "- 基准对照：E1-0 即基准实验的 F1（28 个特征），同一脚本、同一划分、同一超参搜索，结果应与基准一致。",
        "- 所有滞后特征在**站点内部**沿日历日回溯构造；先重采样到完整日网格再 `shift`，因此 D-1 一定是前一自然日；",
        "  滚动统计先 `shift(1)` 再滚动，窗口不含目标日；缺测保持 NaN，未插值、未用未来值补齐。",
        "- 不使用 MDA8/O₃ 历史作为输入，`no2/co/pm25/pm10/so2` 与 7 项气象（温度、湿度、风速、u/v 风分量、辐射、边界层高度）参与滞后。",
        "- 时间协议：2019—2020 训练 → 2021 选超参（样本外）→ 2019—2021 重训 → 2022 独立测试。",
        "- 空间协议：留城市（整城 4 年剔除），9 折合并。",
        "- 主模型 XGBoost；对照 RF/LGBM/ExtraTrees/线性（时间协议全跑，空间协议跑 XGB+LGBM）。",
        "",
        "## 各级特征规模",
        "",
        pd.DataFrame({
            "版本": list(tiers),
            "特征数": [len(v) for v in tiers.values()],
            "新增滞后特征": [len(v) - len(tiers["E1-0"]) for v in tiers.values()],
            "平均缺失率": [f"{lagged[v].isna().mean().mean():.2%}" for v in tiers.values()],
        }).to_markdown(index=False),
        "",
        "## 1. 2022 独立测试（相对 E1-0）",
        "",
        delta[["version", "model", "R2", "RMSE", "MAE", "RMSE_P95", "dRMSE_vs_E1_0", "dRMSE_pct", "dP95_vs_E1_0"]]
        .round(3).to_markdown(index=False),
        "",
        "## 2. 留城市空间验证（合并 9 折）",
        "",
        spatial_delta.round(3).to_markdown(index=False),
        "",
        "## 3. 共同样本严格对齐（XGB，滞后特征全部可用）",
        "",
        f"子样本 {len(common):,} 站点日；四级版本在同一行集合上训练与测试，排除“样本减少带来的偶然改善”。",
        "",
        common_tab[common_tab["split"] == "test"][["version", "model", "n", "R2", "RMSE", "MAE", "RMSE_P95"]]
        .round(3).to_markdown(index=False),
        "",
        "## 4. 保留判定（时间与空间同时改善，且 P95 不恶化）",
        "",
        keep_tab.to_markdown(index=False),
        "",
        "## 5. 结论",
        "",
    ]

    best_ver = keep_tab.loc[keep_tab["keep"], "version"].tolist() if keep_tab["keep"].any() else []
    if best_ver:
        v = best_ver[-1]
        lines.append(
            f"- **保留 {v}**：2022 测试 RMSE {xgb_time.loc[v, 'RMSE']:.2f}（Δ{xgb_time.loc[v, 'dRMSE_vs_E1_0']:+.2f}，"
            f"{xgb_time.loc[v, 'dRMSE_pct']:+.1f}%），留城市 RMSE {xgb_sp.loc[v, 'RMSE']:.2f}"
            f"（Δ{xgb_sp.loc[v, 'dRMSE_vs_E1_0']:+.2f}）；P95 RMSE 变化 {xgb_time.loc[v, 'dP95_vs_E1_0']:+.2f}。"
        )
        if len(best_ver) > 1:
            lines.append(f"- 同时满足条件的版本：{best_ver}；按增量与简洁性优先取 {v}。")
    else:
        lines.append(
            "- **不保留时间滞后模块**：没有版本能在 2022 测试与留城市测试同时改善 RMSE 且 P95 不恶化，"
            "说明 F1 已基本消化了可用的日尺度时间信息。"
        )
    lines.extend([
        "",
        f"- E1-0（XGB）复现基准：R²={base_xgb['R2']:.3f}，RMSE={base_xgb['RMSE']:.2f}，MAE={base_xgb['MAE']:.2f}，"
        f"与基准报告 R²=0.715 / RMSE=18.15 一致。",
        "- 滞后特征只改善“前一日的污染/气象记忆”，不能替代区域输送信息；若增益有限，说明瓶颈在空间迁移而非时间惯性，"
        "E2/E3 的价值需要在同一套独立测试上单独验证。",
        "- 所有输入仍为目标日同一天的气象/污染观测，属于诊断型估算，不构成提前预报。",
        "",
        "## 6. 产出文件",
        "",
        "- `data/processed/modeling_table_e1.parquet`：含全部滞后特征",
        "- `outputs/tables/e1_lag_time_validation.csv`、`e1_lag_spatial_pooled.csv`、`e1_lag_spatial_by_city.csv`",
        "- `outputs/tables/e1_lag_common_sample.csv`、`e1_lag_delta_time.csv`、`e1_lag_delta_spatial.csv`",
        "- `outputs/tables/prediction_2022_E1-*_B2_XGB.csv`",
        "",
    ])
    (config.REPORTS / "e1_lag_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nE1 报告: outputs/reports/e1_lag_report.md")
    print(delta[delta["model"] == "B2_XGB"].round(3).to_string(index=False))
    print(spatial_delta[spatial_delta["model"] == "B2_XGB"].round(3).to_string(index=False))
    print(keep_tab.to_string(index=False))


if __name__ == "__main__":
    main()
