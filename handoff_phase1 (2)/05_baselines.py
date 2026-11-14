"""步骤 5：运行 B0—B4 基准模型，执行时间独立验证与留城市空间验证，输出结果表与报告。"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fjozone import config, models  # noqa: E402


def _plot_time_metrics(time_tab: pd.DataFrame) -> None:
    test = time_tab[time_tab["split"] == "test"].copy()
    groups = test["feature_set"].unique()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, metric in zip(axes, ["R2", "RMSE"]):
        width = 0.8 / len(groups)
        x = np.arange(test["model"].nunique())
        names = sorted(test["model"].unique())
        for i, g in enumerate(groups):
            sub = test[test["feature_set"] == g].set_index("model").reindex(names)
            ax.bar(x + i * width, sub[metric], width, label=g)
        ax.set_xticks(x + width * (len(groups) - 1) / 2)
        ax.set_xticklabels(names, rotation=15)
        ax.set_ylabel(metric)
        ax.set_title(f"2022 独立测试集 {metric}")
        ax.legend(title="特征组")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig_model_comparison.png", dpi=160)
    plt.close(fig)


def _plot_best_scatter(best_pred: pd.DataFrame, label: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(best_pred["y_true"], best_pred["y_pred"], s=6, alpha=0.25, edgecolors="none")
    lim = [0, max(best_pred["y_true"].max(), best_pred["y_pred"].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("观测 MDA8 (μg/m³)")
    ax.set_ylabel("预测 MDA8 (μg/m³)")
    ax.set_title(f"2022 独立测试：{label}")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig_scatter_2022_best.png", dpi=160)
    plt.close(fig)


def _plot_station(best_pred: pd.DataFrame, label: str) -> None:
    st = best_pred.groupby(["city", "station_code"]).apply(
        lambda g: pd.Series(models.metrics(g["y_true"], g["y_pred"])), include_groups=False
    ).reset_index()
    st = st.dropna(subset=["RMSE"])
    st = st.sort_values("RMSE")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(len(st)), st["RMSE"], color="#4C78A8")
    ax.set_xticks(range(len(st)))
    ax.set_xticklabels([f"{c}-{s}" for c, s in zip(st["city"], st["station_code"])], rotation=90, fontsize=7)
    ax.set_ylabel("RMSE (μg/m³)")
    ax.set_title(f"2022 逐站点误差：{label}")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig_station_rmse_2022.png", dpi=160)
    plt.close(fig)
    return st


def main() -> None:
    df = pd.read_parquet(config.DATA_PROCESSED / "modeling_table.parquet")
    zoo = models.model_zoo()

    time_tables, spatial_tables, pooled_tables = [], [], []
    best_pred_by_fs = {}
    importance_rows = []
    params_by_fs = {}

    for fs, feats in models.FEATURE_GROUPS.items():
        if not feats:
            continue
        missing = [f for f in feats if f not in df.columns]
        if missing:
            print(f"[跳过] {fs} 缺少特征列: {missing}")
            continue
        print(f"\n=== 特征组 {fs}（{len(feats)} 个特征）===")
        res = models.run_feature_set(df, feats, zoo)
        tab = res["table"]
        tab.insert(0, "feature_set", fs)
        time_tables.append(tab)
        params_by_fs[fs] = res["params"]

        # 代表模型按 2021 验证集 RMSE 选择（不使用 2022 测试集信息）
        best_name = tab[(tab["split"] == "val")].sort_values("RMSE").iloc[0]["model"]
        best_pred_by_fs[fs] = (res["preds"][(best_name, "test")], best_name)
        best_pred_by_fs[fs][0].to_csv(config.TABLES / f"prediction_2022_{fs}_{best_name}.csv", index=False, encoding="utf-8-sig")

        for name, series in res["importance"].items():
            for feat, val in series.head(15).items():
                importance_rows.append({"feature_set": fs, "model": name, "feature": feat, "importance": float(val)})

        spatial_city, pooled = models.run_spatial(df, feats, zoo, res["params"])
        spatial_city.insert(0, "feature_set", fs)
        pooled.insert(0, "feature_set", fs)
        spatial_tables.append(spatial_city)
        pooled_tables.append(pooled)

    time_tab = pd.concat(time_tables, ignore_index=True)
    spatial_tab = pd.concat(spatial_tables, ignore_index=True)
    pooled_tab = pd.concat(pooled_tables, ignore_index=True)
    importance_tab = pd.DataFrame(importance_rows)

    time_tab.to_csv(config.TABLES / "baseline_time_validation.csv", index=False, encoding="utf-8-sig")
    spatial_tab.to_csv(config.TABLES / "baseline_spatial_validation_by_city.csv", index=False, encoding="utf-8-sig")
    pooled_tab.to_csv(config.TABLES / "baseline_spatial_validation_pooled.csv", index=False, encoding="utf-8-sig")
    importance_tab.to_csv(config.TABLES / "baseline_feature_importance.csv", index=False, encoding="utf-8-sig")

    _plot_time_metrics(time_tab)
    best_fs = [fs for fs in best_pred_by_fs if fs == "F1"] or list(best_pred_by_fs)
    fs_best = best_fs[0]
    pred, model_name = best_pred_by_fs[fs_best]
    _plot_best_scatter(pred, f"{fs_best} / {model_name}")
    st = _plot_station(pred, f"{fs_best} / {model_name}")
    st.to_csv(config.TABLES / f"station_metrics_2022_{fs_best}.csv", index=False, encoding="utf-8-sig")

    # ---- 结论（由数值驱动） ----
    test_tab = time_tab[time_tab["split"] == "test"]
    val_tab = time_tab[time_tab["split"] == "val"]
    interp = []
    consistent = False
    if {"F0", "F1"}.issubset(set(test_tab["feature_set"])):
        d_test = (
            test_tab.pivot_table(index="model", columns="feature_set", values="RMSE")
            .assign(delta=lambda x: x["F1"] - x["F0"])
            .assign(delta_pct=lambda x: x["delta"] / x["F0"] * 100)
        )
        interp.extend([
            "- 2022 独立测试集上，F1 相对 F0 的 RMSE 变化（负值表示污染物特征带来改善）：",
            "",
            d_test.round(2).to_markdown(),
            "",
            f"- 五个模型的平均 ΔRMSE = {d_test['delta'].mean():.2f} μg/m³"
            f"（{d_test['delta_pct'].mean():.1f}%），"
            + ("即常规污染物特征整体带来改善。" if d_test["delta"].mean() < 0 else "即常规污染物特征未带来整体改善。"),
            "",
        ])
    if {"F0", "F1"}.issubset(set(pooled_tab["feature_set"])):
        d_sp = (
            pooled_tab.pivot_table(index="model", columns="feature_set", values="RMSE")
            .assign(delta=lambda x: x["F1"] - x["F0"])
            .assign(delta_pct=lambda x: x["delta"] / x["F0"] * 100)
        )
        interp.extend([
            "- 留城市空间验证（合并 9 折）上，F1 相对 F0 的 RMSE 变化：",
            "",
            d_sp.round(2).to_markdown(),
            "",
            f"- 空间验证中平均 ΔRMSE = {d_sp['delta'].mean():.2f} μg/m³（{d_sp['delta_pct'].mean():.1f}%）。",
            "",
        ])
        consistent = test_tab.pivot_table(index="model", columns="feature_set", values="RMSE")
        consistent = (consistent["F1"] - consistent["F0"]).mean() < 0 and d_sp["delta"].mean() < 0
        interp.append(
            "- 结论：**常规污染物（F1）在时间独立与空间留出两种协议下"
            + ("均带来稳定增益**，可保留为基准特征组。" if consistent else
               "未同时带来稳定增益**，基准模型暂以 F0 为主，污染物特征仅作为对照。")
        )
        interp.append("")
    else:
        interp.append("- 仅运行了 F0，未比较污染物特征。")
        interp.append("")

    # 训练→验证→测试的泛化落差
    if not val_tab.empty:
        gap = (
            test_tab.groupby("feature_set")["RMSE"].mean()
            - val_tab.groupby("feature_set")["RMSE"].mean()
        )
        interp.extend([
            "### 泛化落差",
            "",
            "2022 测试集与 2021 验证集的平均 RMSE 差异（正值表示独立年份变差）：",
            "",
            gap.round(2).to_frame("RMSE_gap_2022_minus_2021").to_markdown(),
            "",
            "若落差明显，说明模型对未见年份的外推能力有限，第二阶段模块应以改善独立年份表现为目标。",
            "",
        ])

    best_test = test_tab.loc[test_tab["RMSE"].idxmin()]
    interp.extend([
        "### 最佳基准模型",
        "",
        f"2022 独立测试集 RMSE 最低的组合为 **{best_test['feature_set']} / {best_test['model']}**"
        f"（R²={best_test['R2']:.3f}，RMSE={best_test['RMSE']:.2f}，MAE={best_test['MAE']:.2f} μg/m³）。",
        "",
    ])

    # ---- 报告 ----
    lines = [
        "# 福建省 O₃ MDA8 基准实验报告（2019—2022）",
        "",
        "## 验证协议",
        "",
        "- 时间独立：2019—2020 训练 → 2021 验证（模型仅用 2019—2020 拟合，用于选超参数）→ 2019—2021 重训 → **2022 独立测试**（不参与调参）。",
        "- 空间独立：**留城市**（整座城市 4 年数据全部剔除），训练集不含测试城市任何站点/任何年份，避免空间泄漏。",
        "- 相同样本、相同划分、相同特征，仅模型不同；五个模型共享同一套超参数搜索流程。",
        "",
        "## 1. 时间独立验证（2022 测试集）",
        "",
        time_tab[time_tab["split"] == "test"][
            ["feature_set", "model", "n", "R2", "RMSE", "MAE", "RMSE_P90", "MAE_P90", "RMSE_P95", "MAE_P95"]
        ].round(3).to_markdown(index=False),
        "",
        "验证年（2021，仅用 2019—2020 训练）结果：",
        "",
        time_tab[time_tab["split"] == "val"][["feature_set", "model", "n", "R2", "RMSE", "MAE"]].round(3).to_markdown(index=False),
        "",
        "## 2. 空间独立验证（留城市，合并全部折）",
        "",
        pooled_tab[["feature_set", "model", "n", "R2", "RMSE", "MAE", "RMSE_P90", "RMSE_P95"]].round(3).to_markdown(index=False),
        "",
        "各城市明细见 `outputs/tables/baseline_spatial_validation_by_city.csv`。",
        "",
        "## 3. 逐站点误差（2022，代表模型）",
        "",
        st.sort_values("RMSE")[["city", "station_code", "n", "R2", "RMSE", "MAE"]].round(3).to_markdown(index=False),
        "",
        "## 4. 特征重要性（各模型前 15 位）",
        "",
        importance_tab.groupby(["feature_set", "model"]).head(8)[["feature_set", "model", "feature", "importance"]]
        .round(4).to_markdown(index=False),
        "",
        "## 5. 选定的超参数",
        "",
        "```",
        *[f"{fs}: {p}" for fs, p in params_by_fs.items()],
        "```",
        "",
        "## 6. 结果解读",
        "",
        *interp,
        "## 7. 研究边界与下一步",
        "",
        "- F2（TROPOMI 卫星柱浓度）尚未接入，`models.FEATURE_GROUPS` 中仅预留接口，本报告结论只覆盖 F0/F1。",
        "- 模型使用的是同日气象与污染物输入，属于**估算/诊断**而非提前预报；不得表述为预报能力。",
        "- 2022 为独立年份测试，2021 仅用于选超参数，未参与最终训练之外的调参决策。",
        "- 第二阶段模块（轨迹加权、化学敏感性代理、动态集成、高臭氧加权）需以上述独立测试与留城市结果为门槛，"
        "只有在两种协议下均稳定改善才保留。",
        "",
    ]
    (config.REPORTS / "baseline_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n结果表与报告已输出到 {config.TABLES} / {config.REPORTS}")


if __name__ == "__main__":
    main()
