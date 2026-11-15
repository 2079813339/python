"""步骤 10：嵌套留城市验证（修正 E1 阶段前空间验证的超参数泄漏）。

原协议先用全部城市 2021 年数据选超参，再在留城市折中训练；被留城市的 2021 年数据
虽未进入拟合，却参与了超参数选择。本脚本改为每个外层折内部独立调参：

  内层训练 = 其余 8 城 2019—2020；内层验证 = 其余 8 城 2021；
  外层训练 = 其余 8 城 2019—2022；外层测试 = 被留城市 2019—2022。

同时把 E1 的四级滞后版本在新协议下重算，保证第二阶段的对照基准一致。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fjozone import config, lagfeatures, models  # noqa: E402


def main() -> None:
    base = pd.read_parquet(config.DATA_PROCESSED / "modeling_table.parquet")
    zoo = models.model_zoo()

    summary = []
    for name in ("F0", "F1"):
        feats = models.FEATURE_GROUPS[name]
        print(f"\n=== 嵌套留城市：{name}（{len(feats)} 特征，5 模型）===", flush=True)
        per_city, pooled, params = models.run_spatial_nested(base, feats, zoo, return_params=True)
        per_city.insert(0, "feature_set", name)
        pooled.insert(0, "feature_set", name)
        params.insert(0, "feature_set", name)
        per_city.to_csv(config.TABLES / f"spatial_nested_by_city_{name}.csv", index=False, encoding="utf-8-sig")
        pooled.to_csv(config.TABLES / f"spatial_nested_pooled_{name}.csv", index=False, encoding="utf-8-sig")
        params.to_csv(config.TABLES / f"spatial_nested_params_{name}.csv", index=False, encoding="utf-8-sig")
        summary.append(pooled)
        print(pooled.round(3).to_string(index=False), flush=True)
        print(params.to_string(index=False), flush=True)

    pooled_base = pd.concat(summary, ignore_index=True)
    pooled_base.to_csv(config.TABLES / "spatial_nested_pooled.csv", index=False, encoding="utf-8-sig")

    # 旧协议（2021 全城选参）对照
    old_path = config.TABLES / "baseline_spatial_pooled.csv"
    old = pd.read_csv(old_path) if old_path.exists() else None

    # E1 四级版本在新协议下重算（XGB/LGBM）
    lagged = pd.read_parquet(config.DATA_PROCESSED / "modeling_table_e1.parquet")
    zoo_sp = {k: zoo[k] for k in ("B2_XGB", "B3_LGBM")}
    e1_pooled = []
    for ver, feats in lagfeatures.feature_tiers().items():
        print(f"\n=== 嵌套留城市：{ver}（{len(feats)} 特征）===", flush=True)
        per_city, pooled_e1 = models.run_spatial_nested(lagged, feats, zoo_sp)
        per_city.insert(0, "version", ver)
        pooled_e1.insert(0, "version", ver)
        per_city.to_csv(config.TABLES / f"e1_lag_spatial_nested_by_city_{ver}.csv", index=False, encoding="utf-8-sig")
        e1_pooled.append(pooled_e1)
        print(pooled_e1.round(3).to_string(index=False), flush=True)
    e1_tab = pd.concat(e1_pooled, ignore_index=True)
    e1_tab.to_csv(config.TABLES / "e1_lag_spatial_nested_pooled.csv", index=False, encoding="utf-8-sig")

    # ---- 汇总报告 ----
    lines = [
        "# 嵌套留城市验证结果（修正超参数泄漏）",
        "",
        "## 修正内容",
        "",
        "| 项目 | 原协议 | 新协议（嵌套） |",
        "|:-----|:-------|:---------------|",
        "| 超参选择数据 | 全部 9 城 2019—2020 训练 + 全部 9 城 2021 验证 | 其余 8 城 2019—2020 训练 + 其余 8 城 2021 验证 |",
        "| 外层训练 | 其余 8 城 2019—2022 | 其余 8 城 2019—2022 |",
        "| 外层测试 | 被留城市 2019—2022 | 被留城市 2019—2022 |",
        "| 被留城市是否影响调参 | **是** | 否 |",
        "",
        "## 基准特征组（9 折合并，5 模型）",
        "",
        pooled_base.round(3).to_markdown(index=False),
        "",
        "### 各折选出的超参数",
        "",
        params.round(3).to_markdown(index=False),
        "",
    ]
    if old is not None:
        cmp = old.merge(pooled_base, on=["feature_set", "model"], suffixes=("_old", "_nested"))
        cmp["dRMSE"] = cmp["RMSE_nested"] - cmp["RMSE_old"]
        lines += [
            "## 与原协议对照（RMSE）",
            "",
            cmp[["feature_set", "model", "RMSE_old", "RMSE_nested", "dRMSE",
                 "R2_old", "R2_nested"]].round(3).to_markdown(index=False),
            "",
            "正值表示原协议因调参泄漏而偏乐观。",
            "",
        ]
    lines += [
        "## E1 四级版本（嵌套协议，XGB/LGBM）",
        "",
        e1_tab.round(3).to_markdown(index=False),
        "",
        "## 解读要点",
        "",
        "- 2022 时间独立测试不受此次修正影响（2022 从未参与调参），仍沿用 F1/XGB RMSE≈18.15 的基准。",
        "- 空间结果以本报告为准；此前 baseline_report.md 与 e1_lag_report.md 中的留城市数值为旧协议，",
        "  仅保留作对照，不再作为模块保留依据。",
        "",
    ]
    (config.REPORTS / "spatial_nested_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n报告: outputs/reports/spatial_nested_report.md")


if __name__ == "__main__":
    main()
