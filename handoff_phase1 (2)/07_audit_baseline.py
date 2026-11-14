"""步骤 7：第二阶段开工前的基准实验审计。

逐项核查：样本完整性、重复记录、站点编码、训练/验证/测试日期边界、特征时间戳与泄漏、
以及用已保存的 2022 预测文件复现报告中的 RMSE，确认第二阶段实验可以挂在同一套划分上。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fjozone import config, models  # noqa: E402

LABEL_COLS = {"mda8", "mda8_raw", "o3_mean", "o3_8h", "o3_mda8_official", "o3_mean24", "day_valid"}
OUTCOMES = []


def check(name: str, ok: bool, detail: str, warn_only: bool = False) -> None:
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    OUTCOMES.append({"检查项": name, "结果": status, "说明": detail})
    print(f"[{status}] {name}: {detail}", flush=True)


def main() -> None:
    df = pd.read_parquet(config.DATA_PROCESSED / "modeling_table.parquet")
    print(f"建模表: {df.shape[0]:,} 行 × {df.shape[1]} 列\n")

    # ---- 1. 基本结构与标签 ----
    required = {"station_code", "city", "date", "mda8", "year"}
    check("必需列", required.issubset(df.columns), f"缺失 {sorted(required - set(df.columns))}")
    check("标签无缺失", df["mda8"].notna().all(), f"mda8 缺失 {int(df['mda8'].isna().sum())} 行")
    check("date 为日尺度", (df["date"].dt.hour == 0).all() and (df["date"].dt.minute == 0).all(),
          f"非零时刻 {int(((df['date'].dt.hour != 0) | (df['date'].dt.minute != 0)).sum())} 行")
    check("日期范围", str(df["date"].min().date()) >= "2019-01-01" and str(df["date"].max().date()) <= "2022-12-31",
          f"{df['date'].min().date()} ~ {df['date'].max().date()}")
    check("year 与 date 一致", (df["date"].dt.year == df["year"]).all(), "year 列与 date 年份不符的行数为 "
          f"{int((df['date'].dt.year != df['year']).sum())}")

    # ---- 2. 重复记录与键唯一性 ----
    dup = int(df.duplicated(["station_code", "date"]).sum())
    check("站点—日期唯一", dup == 0, f"重复 {dup} 行")
    bad_code = ~df["station_code"].astype(str).str.fullmatch(r"\d+A")
    check("站点编码格式", not bad_code.any(), f"异常编码 {df.loc[bad_code, 'station_code'].unique()[:5].tolist()}")
    check("站点编码无空白", df["station_code"].astype(str).str.strip().eq(df["station_code"].astype(str)).all(),
          "编码两侧存在空格时会导致训练/测试站点错配")
    check("城市归属唯一", (df.groupby("station_code")["city"].nunique() == 1).all(),
          f"跨城市站点 {int((df.groupby('station_code')['city'].nunique() > 1).sum())} 个")

    # ---- 3. 划分边界 ----
    years = df["year"]
    train = df[years.isin(config.TRAIN_YEARS)]
    val = df[years == config.VAL_YEAR]
    test = df[years == config.TEST_YEAR]
    check("年份三分", len(train) + len(val) + len(test) == len(df),
          f"train={len(train):,} val={len(val):,} test={len(test):,}")
    check("测试年仅 2022", set(test["year"].astype(int).unique()) == {config.TEST_YEAR},
          f"{sorted(int(y) for y in test['year'].unique())}")
    check("验证年仅 2021", set(val["year"].astype(int).unique()) == {config.VAL_YEAR},
          f"{sorted(int(y) for y in val['year'].unique())}")
    check("2021 不进训练集", not set(train["date"]).intersection(set(val["date"])),
          "训练/验证日期交集为空")
    check("2022 不进调参", not set(train["date"]).intersection(set(test["date"])), "训练/测试日期交集为空")

    # ---- 4. 特征时间戳与泄漏 ----
    f1 = models.FEATURE_GROUPS["F1"]
    leaked = [c for c in f1 if c in LABEL_COLS]
    check("F1 无标签泄漏列", not leaked, f"疑似泄漏特征 {leaked}")
    o3_cols = [c for c in df.columns if "o3" in c.lower()]
    check("F1 不含任何 O3 派生列", not [c for c in f1 if c in o3_cols],
          f"建模表中 O3 相关列 {o3_cols} 均未进入特征")
    check("F1 特征齐全", all(c in df.columns for c in f1),
          f"缺失 {[c for c in f1 if c not in df.columns]}")
    check("气象为同日日尺度聚合",
          all(c.startswith(("t2m", "dew", "rh", "ws", "u_", "v_", "srad", "blh", "precip", "sp_", "cloud", "vpd", "soil"))
              for c in models.METEO_FEATURES),
          "16 个气象特征均为目标日 00—24 时的日均/日累/日极值，属于同日诊断输入，未使用未来日期信息")
    check("特征无滞后结构（基准阶段）", not any(("lag" in c.lower() or "roll" in c.lower()) for c in f1),
          "基准 F1 不含任何滞后/滚动特征，E1 将在独立脚本中新增")

    # ---- 5. 缺失与覆盖 ----
    miss = df[f1].isna().mean().sort_values(ascending=False)
    check("特征缺失率", miss.max() < 0.10, f"最高缺失率 {miss.index[0]} {miss.iloc[0]:.2%}，"
          f"均值 {miss.mean():.2%}")
    per_year = df.groupby("year").agg(n=("mda8", "size"), stations=("station_code", "nunique"),
                                      mda8_mean=("mda8", "mean")).round(2)
    print("\n逐年样本：")
    print(per_year.to_string(), flush=True)

    # ---- 6. 用已保存预测复现 2022 指标 ----
    pred_path = config.TABLES / "prediction_2022_F1_B2_XGB.csv"
    if pred_path.exists():
        pred = pd.read_csv(pred_path, parse_dates=["date"])
        m = models.metrics(pred["y_true"], pred["y_pred"])
        check("2022 预测样本数", len(pred) == len(test), f"预测 {len(pred):,} 行 / 测试集 {len(test):,} 行")
        check("2022 站点覆盖", set(pred["station_code"]) == set(test["station_code"]),
              f"{pred['station_code'].nunique()} 个站点")
        check("指标可复现（R²）", abs(m["R2"] - 0.715) < 0.005, f"复算 R²={m['R2']:.4f}（报告 0.715）")
        check("指标可复现（RMSE）", abs(m["RMSE"] - 18.15) < 0.05, f"复算 RMSE={m['RMSE']:.3f}（报告 18.152）")
        check("指标可复现（MAE）", abs(m["MAE"] - 13.98) < 0.05, f"复算 MAE={m['MAE']:.3f}（报告 13.98）")
        check("指标可复现（P95）", abs(m["RMSE_P95"] - 36.55) < 0.10, f"复算 P95 RMSE={m['RMSE_P95']:.3f}（报告 36.548）")
        # 预测与标签逐行核对
        joined = pred.merge(test[["station_code", "date", "mda8"]], on=["station_code", "date"], how="inner")
        max_diff = float(np.nanmax(np.abs(joined["y_true"] - joined["mda8"]))) if len(joined) else np.nan
        check("预测行与标签对齐", len(joined) == len(pred) and max_diff < 1e-6,
              f"对齐 {len(joined):,} 行，y_true 与建模表 mda8 最大差 {max_diff:.2e}")
    else:
        check("2022 预测文件", False, f"未找到 {pred_path.name}")

    # ---- 7. 空间划分可行性 ----
    cities = sorted(df["city"].unique())
    cov = df.groupby("city").agg(stations=("station_code", "nunique"), n=("mda8", "size"),
                                 years=("year", "nunique"))
    check("留城市折数", len(cities) == 9, f"{cities}")
    check("每座城市均覆盖 4 年", (cov["years"] == 4).all(),
          f"不足 4 年: {cov.index[cov['years'] < 4].tolist()}")
    print("\n城市覆盖：")
    print(cov.to_string(), flush=True)

    audit = pd.DataFrame(OUTCOMES)
    n_fail = int((audit["结果"] == "FAIL").sum())
    n_warn = int((audit["结果"] == "WARN").sum())

    lines = [
        "# 基准实验审计报告（第二阶段前置检查）",
        "",
        f"- 建模表：`data/processed/modeling_table.parquet`，{len(df):,} 行 × {df.shape[1]} 列",
        f"- 站点 {df['station_code'].nunique()} 个，城市 {df['city'].nunique()} 个，年份 {df['year'].min()}—{df['year'].max()}",
        f"- 审计结论：{len(audit)} 项检查，PASS {len(audit) - n_fail - n_warn}，WARN {n_warn}，FAIL {n_fail}",
        "",
        "## 检查明细",
        "",
        audit.to_markdown(index=False),
        "",
        "## 逐年样本",
        "",
        per_year.to_markdown(),
        "",
        "## 城市覆盖（留城市折）",
        "",
        cov.to_markdown(),
        "",
        "## 审计口径说明",
        "",
        "- 时间戳：CNEMC 小时观测按北京时间发布，MDA8 按北京时间自然日 00—24 时滑动窗口计算，",
        "  因此站点日期与气象日值（Open-Meteo `timezone=Asia/Shanghai`）为同一自然日，无跨时区错位。",
        "- 气象特征为目标日全天聚合量（日均/日累/日极值），属于同日诊断型输入；",
        "  本阶段不声称任何提前预报能力。",
        "- 特征中不含 O₃ 及其派生量（`o3_mean`、`o3_8h`、官方 MDA8 等），污染物仅保留 NO₂/CO/PM₂.₅/PM₁₀/SO₂。",
        "- 留城市验证按整城剔除 4 年全部站点日，训练集不含该城任何年份记录，无空间泄漏。",
        "",
    ]
    (config.REPORTS / "baseline_audit.md").write_text("\n".join(lines), encoding="utf-8")
    audit.to_csv(config.TABLES / "baseline_audit_checks.csv", index=False, encoding="utf-8-sig")
    print(f"\n审计报告: {config.REPORTS / 'baseline_audit.md'}")
    if n_fail:
        print("存在 FAIL 项，请先修复再进入 E1。")


if __name__ == "__main__":
    main()
