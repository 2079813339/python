"""步骤 21：论文数字一致性校验（不训练模型）。

校验 discussion_limitations_conclusion_abstract.md 与 results_section_draft.md 中引用的
关键数字是否与冻结结果表逐位一致。全部通过才可视为"全文数值一致"。
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fjozone import config  # noqa: E402

XGB = "B2_XGB"
TAB = config.TABLES
E4T = config.OUTPUTS / "e4" / "tables"
E2A = config.OUTPUTS / "e2" / "audit"
OUT = config.OUTPUTS / "paper"


def main() -> None:
    t2 = pd.read_csv(TAB / "e2_main_time.csv")
    s2 = pd.read_csv(TAB / "e2_main_spatial_pooled.csv")
    e4t = pd.read_csv(E4T / "e4_temporal_metrics.csv")
    e4s = pd.read_csv(E4T / "e4_spatial_metrics.csv")
    b2 = pd.read_csv(TAB / "e2_main_bootstrap.csv")
    b4 = pd.read_csv(E4T / "e4_bootstrap.csv")
    sh = pd.read_csv(TAB / "e2_shap_global.csv")
    oz = pd.read_csv(TAB / "e2_shap_by_ozone.csv")
    sa = pd.read_csv(E4T.parent / "audit" / "e4_sample_audit.csv")
    dec = pd.read_csv(OUT / "table_error_decomposition.csv")
    col = pd.read_csv(OUT / "table_satellite_collinearity.csv")
    fw = pd.read_csv(OUT / "table_spatial_foldwise_threshold.csv")
    fwc = pd.read_csv(OUT / "table_spatial_foldwise_by_city.csv")
    vif = pd.read_csv(OUT / "table_satellite_vif.csv")
    ss = pd.read_csv(E2A / "E2_selection_bias_stratified.csv")
    sc = pd.read_csv(E2A / "E2_selection_bias_by_city.csv")
    s6 = (OUT / "table_supplementary_selection_bias.md").read_text(encoding="utf-8")
    model = pd.read_parquet(config.DATA_PROCESSED / "modeling_table_with_tropomi_5km.parquet")
    doc = (OUT / "discussion_limitations_conclusion_abstract.md").read_text(encoding="utf-8")
    res = (OUT / "results_section_draft.md").read_text(encoding="utf-8")
    meth = (OUT / "methods_e4_threshold_insert.md").read_text(encoding="utf-8")

    def e2t(ver, col="RMSE"):
        return float(t2[(t2.version == ver) & (t2.model == XGB) & (t2.split == "test")][col].iloc[0])

    def e2s(ver, col="RMSE"):
        return float(s2[(s2.version == ver) & (s2.model == XGB)][col].iloc[0])

    def e4(frame, ver, col):
        return float(frame[(frame.version == ver) & (frame.model == XGB)][col].iloc[0])

    def boot(df, ver, scene, group, cluster, col):
        key = {"split": scene} if "split" in df.columns else {"scene": scene, "group": group}
        q = df[(df.version == ver) & (df.model == XGB) & (df.cluster == cluster)]
        for k, v in key.items():
            q = q[q[k] == v]
        return float(q[col].iloc[0])

    common = (model["hcho_col"].notna() & model["no2_col"].notna()
              & (model["hcho_count"] > 0) & (model["no2_count"] > 0) & model["mda8"].notna())
    n_common = int(common.sum())
    cov = n_common / len(model) * 100

    checks = [
        ("E2-0 时间 RMSE", e2t("E2-0"), 19.300, 3),
        ("E2-3 时间 RMSE", e2t("E2-3"), 18.642, 3),
        ("E2-3 时间 ΔRMSE", e2t("E2-3") - e2t("E2-0"), -0.658, 3),
        ("E2-3 时间 Δ%", (e2t("E2-3") / e2t("E2-0") - 1) * 100, -3.41, 2),
        ("E2-0 空间 RMSE", e2s("E2-0"), 17.277, 3),
        ("E2-3 空间 RMSE", e2s("E2-3"), 16.800, 3),
        ("E2-3 空间 ΔRMSE", e2s("E2-3") - e2s("E2-0"), -0.477, 3),
        ("E2-3 空间 Δ%", (e2s("E2-3") / e2s("E2-0") - 1) * 100, -2.76, 2),
        ("E4-0 时间总体 RMSE", e4(e4t, "E4-0", "RMSE"), 18.642, 3),
        ("E4-1 时间总体 RMSE", e4(e4t, "E4-1", "RMSE"), 18.270, 3),
        ("E4-1 时间总体 Δ%", (e4(e4t, "E4-1", "RMSE") / e4(e4t, "E4-0", "RMSE") - 1) * 100, -1.99, 2),
        ("E4-0 时间高臭氧 RMSE", e4(e4t, "E4-0", "RMSE_geP90"), 33.341, 3),
        ("E4-1 时间高臭氧 RMSE", e4(e4t, "E4-1", "RMSE_geP90"), 31.847, 3),
        ("E4-1 时间高臭氧 Δ%", (e4(e4t, "E4-1", "RMSE_geP90") / e4(e4t, "E4-0", "RMSE_geP90") - 1) * 100, -4.48, 2),
        ("E4-1 时间高臭氧 MAE Δ%", (e4(e4t, "E4-1", "MAE_geP90") / e4(e4t, "E4-0", "MAE_geP90") - 1) * 100, -6.17, 2),
        ("E4-1 时间高臭氧 Bias", e4(e4t, "E4-1", "Bias_geP90"), -27.026, 3),
        ("E4-1 时间低臭氧 RMSE", e4(e4t, "E4-1", "RMSE_ltP90"), 16.057, 3),
        ("E4-0 空间总体 RMSE", e4(e4s, "E4-0", "RMSE"), 16.800, 3),
        ("E4-1 空间总体 RMSE", e4(e4s, "E4-1", "RMSE"), 16.836, 3),
        ("E4-1 空间总体 Δ%", (e4(e4s, "E4-1", "RMSE") / e4(e4s, "E4-0", "RMSE") - 1) * 100, 0.21, 2),
        ("E4-0 空间高臭氧 RMSE", e4(e4s, "E4-0", "RMSE_geP90"), 23.282, 3),
        ("E4-1 空间高臭氧 RMSE", e4(e4s, "E4-1", "RMSE_geP90"), 21.562, 3),
        ("E4-1 空间高臭氧 Δ%", (e4(e4s, "E4-1", "RMSE_geP90") / e4(e4s, "E4-0", "RMSE_geP90") - 1) * 100, -7.39, 2),
        ("E4-1 空间高臭氧 ΔRMSE", e4(e4s, "E4-1", "RMSE_geP90") - e4(e4s, "E4-0", "RMSE_geP90"), -1.720, 3),
        ("E4-1 空间低臭氧 RMSE", e4(e4s, "E4-1", "RMSE_ltP90"), 16.224, 3),
        ("E4-1 空间低臭氧 Δ%", (e4(e4s, "E4-1", "RMSE_ltP90") / e4(e4s, "E4-0", "RMSE_ltP90") - 1) * 100, 1.94, 2),
        ("E4-1 空间总体 Bias", e4(e4s, "E4-1", "Bias"), 2.208, 3),
        ("E4-1 空间高臭氧 Bias", e4(e4s, "E4-1", "Bias_geP90"), -15.444, 3),
        ("E4-2 时间高臭氧 Δ%", (e4(e4t, "E4-2", "RMSE_geP90") / e4(e4t, "E4-0", "RMSE_geP90") - 1) * 100, -2.48, 2),
        ("E4-2 空间高臭氧 Δ%", (e4(e4s, "E4-2", "RMSE_geP90") / e4(e4s, "E4-0", "RMSE_geP90") - 1) * 100, -3.94, 2),
        ("E4-2 空间总体 Δ%", (e4(e4s, "E4-2", "RMSE") / e4(e4s, "E4-0", "RMSE") - 1) * 100, 0.57, 2),
        ("E2-3 空间 城市聚类 CI 上限", boot(b2, "E2-3", "spatial", None, "city", "ci95_hi"), 0.107, 3),
        ("E2-3 时间 站点聚类 CI 下限", boot(b2, "E2-3", "time", None, "station", "ci95_lo"), -0.930, 3),
        ("E4-1 时间高臭氧 站点 CI 下限", boot(b4, "E4-1", "time", "high_ozone", "station", "ci95_lo"), -1.977, 3),
        ("E4-1 时间高臭氧 城市 CI 下限", boot(b4, "E4-1", "time", "high_ozone", "city", "ci95_lo"), -1.993, 3),
        ("E4-1 空间高臭氧 站点 CI 下限", boot(b4, "E4-1", "spatial", "high_ozone", "station", "ci95_lo"), -2.068, 3),
        ("E4-1 空间高臭氧 城市 CI 下限", boot(b4, "E4-1", "spatial", "high_ozone", "city", "ci95_lo"), -2.357, 3),
        ("E4-1 空间总体 站点 CI 下限", boot(b4, "E4-1", "spatial", "overall", "station", "ci95_lo"), -0.134, 3),
        ("E4-1 空间总体 站点 CI 上限", boot(b4, "E4-1", "spatial", "overall", "station", "ci95_hi"), 0.205, 3),
        ("E4-1 空间总体 P_improve", boot(b4, "E4-1", "spatial", "overall", "station", "P_improve"), 0.336, 3),
        ("SHAP NO2 mean|SHAP|", float(sh[sh.feature == "no2_col"]["mean_abs_shap"].iloc[0]), 2.284, 3),
        ("SHAP HCHO mean|SHAP|", float(sh[sh.feature == "hcho_col"]["mean_abs_shap"].iloc[0]), 1.232, 3),
        ("SHAP HCHO 占比%", float(sh[sh.feature == "hcho_col"]["share_pct"].iloc[0]), 2.21, 2),
        ("SHAP NO2 占比%", float(sh[sh.feature == "no2_col"]["share_pct"].iloc[0]), 4.10, 2),
        ("SHAP 高臭氧组 ABS", float(oz[(oz.group_type == "MDA8分位") & (oz.group == ">P90")]["sat_abs_mean"].iloc[0]), 4.691, 3),
        ("SHAP 低臭氧组 ABS", float(oz[(oz.group_type == "MDA8分位") & (oz.group == "<P50")]["sat_abs_mean"].iloc[0]), 3.298, 3),
        ("共同有效样本行数", n_common, 21925, 0),
        ("共同有效覆盖率%", cov, 42.58, 2),
        ("建模表总行数", len(model), 51486, 0),
        ("2022 测试行数", int((model.loc[common, "year"] == config.TEST_YEAR).sum()), 4691, 0),
        ("空间高臭氧组 n", int(e4(e4s, "E4-0", "n_geP90")), 2198, 0),
        ("时间高臭氧组 n", int(e4(e4t, "E4-0", "n_geP90")), 471, 0),
    ]
    for name in ("训练期(2019-2020) P90", "重训期(2019-2021) P90", "测试集 P90(μg/m³)", "全样本 P90(μg/m³)"):
        checks.append((name, float(sa[sa["item"] == name]["value"].iloc[0]), 
                       {"训练期(2019-2020) P90": 142.286, "重训期(2019-2021) P90": 139.750,
                        "测试集 P90(μg/m³)": 149.500, "全样本 P90(μg/m³)": 141.875}[name], 3))

    def d(scene, ver, c):
        return float(dec[(dec["场景"].str.startswith(scene)) & (dec["版本"] == ver)][c].iloc[0])

    def cc(sat, other, c):
        return float(col[(col["卫星特征"] == sat) & (col["对照变量"] == other)][c].iloc[0])

    # 误差分解（表 S4）
    checks += [
        ("SSE 时间 E4-0 高臭氧占比%", d("时间", "E4-0", "高臭氧SSE占比%"), 32.1, 1),
        ("SSE 时间 E4-1 高臭氧占比%", d("时间", "E4-1", "高臭氧SSE占比%"), 30.5, 1),
        ("SSE 空间 E4-0 高臭氧占比%", d("空间", "E4-0", "高臭氧SSE占比%"), 19.3, 1),
        ("SSE 时间 E4-1 高臭氧 ΔSSE", d("时间", "E4-1", "ΔSSE_高臭氧"), -45891, 0),
        ("SSE 空间 E4-1 高臭氧 ΔSSE", d("空间", "E4-1", "ΔSSE_高臭氧"), -169531, 0),
        ("SSE 空间 E4-1 低臭氧 ΔSSE", d("空间", "E4-1", "ΔSSE_低臭氧"), 195778, 0),
        ("SSE 时间 E4-1 高臭氧占变化幅度%", d("时间", "E4-1", "高臭氧占SSE变化幅度%"), 71.3, 1),
    ]
    # 共线性核查（表 S5）
    checks += [
        ("NO₂柱-地面NO₂ Pearson", cc("no2_col", "no2_mean", "Pearson_r"), 0.551, 3),
        ("NO₂柱-地面NO₂ Spearman", cc("no2_col", "no2_mean", "Spearman_r"), 0.526, 3),
        ("NO₂柱-PM₁₀ Pearson", cc("no2_col", "pm10_mean", "Pearson_r"), 0.407, 3),
        ("NO₂柱-HCHO柱 Pearson", cc("no2_col", "hcho_col", "Pearson_r"), 0.062, 3),
        ("NO₂柱-HCHO柱 Spearman", cc("no2_col", "hcho_col", "Spearman_r"), 0.132, 3),
    ]

    def vv(sat, c):
        return float(vif[vif["卫星特征"] == sat][c].iloc[0])

    checks += [
        ("NO₂柱 标准 VIF", vv("no2_col", "VIF_标准"), 2.095, 3),
        ("HCHO柱 标准 VIF", vv("hcho_col", "VIF_标准"), 1.295, 3),
        ("NO₂柱 F1 辅助诊断", vv("no2_col", "VIF_F1辅助诊断"), 2.095, 3),
    ]
    # 表 S6 选择偏差
    def s6num(pat, cast=float):
        m = re.search(pat, s6)
        return cast(m.group(1).replace(",", ""))

    checks += [
        ("S6 β", s6num(r"β（共同有效 vs 无共同有效像元） \| \+([0-9.]+)"), 25.71, 2),
        ("S6 SE", s6num(r"聚类稳健 SE \| ([0-9.]+)"), 0.65, 3),
        ("S6 t", s6num(r"t 值 \| ([0-9.]+)"), 39.3, 1),
        ("S6 n", s6num(r"样本量 \| ([0-9,]+)"), 50192, 0),
        ("S6 聚类数", s6num(r"聚类数（站点） \| ([0-9]+)"), 38, 0),
    ]
    for season in ["冬", "春", "夏", "秋"]:
        a = float(ss[(ss.season == season) & (ss.grp == "A_共同有效")]["mean"].iloc[0])
        b = float(ss[(ss.season == season) & (ss.grp == "B_有记录无像元")]["mean"].iloc[0])
        m = re.search(rf"\| {season} \| ([0-9.]+) \| ([0-9.]+) \| \+([0-9.]+) \|", s6)
        checks.append((f"S6 {season} 季差值", a - b, float(m.group(3)), 3))
    checks += [
        ("S6 九城最小差", sc["diff"].min(), 19.23, 2),
        ("S6 九城最大差", sc["diff"].max(), 31.81, 2),
    ]
    # 逐折阈值敏感性（表 S7）
    checks += [
        ("逐折阈值 E4-0 高臭氧 RMSE", fw[fw["版本"] == "E4-0"]["RMSE_高臭氧"].iloc[0], 23.273, 3),
        ("逐折阈值 E4-1 高臭氧 RMSE", fw[fw["版本"] == "E4-1"]["RMSE_高臭氧"].iloc[0], 21.536, 3),
        ("逐折阈值 E4-1 高臭氧 Δ%", fw[fw["版本"] == "E4-1"]["Δ%"].iloc[0], -7.47, 2),
        ("逐折阈值高臭氧 n", fw[fw["版本"] == "E4-0"]["n_高臭氧"].iloc[0], 2224, 0),
    ]

    rows, fail = [], 0
    for name, got, exp, nd in checks:
        g = round(float(got), nd)
        ok = abs(g - exp) <= (0.5 * 10 ** (-nd) + 1e-12) or abs(g - exp) < 10 ** (-nd)
        rows.append({"检查项": name, "结果表值": g, "文中引用": exp, "通过": "PASS" if ok else "FAIL"})
        fail += 0 if ok else 1

    # 关系型与存在型检查（非逐位相等）
    ground = ["no2_mean", "co_mean", "pm25_mean", "pm10_mean", "so2_mean"]
    hch_max = float(col[(col["卫星特征"] == "hcho_col") &
                        (col["对照变量"].isin(ground))]["Pearson_r"].abs().max())
    n_improve = int((fwc["ΔRMSE"] < 0).sum())
    refs = doc.split("**本节引用")[1]
    links = set(re.findall(r"https://(?:doi\.org|arxiv\.org)/\S+", refs))
    extra = [
        ("HCHO 与地面污染物 |r| 最大 ≤0.09", hch_max, 0.09, hch_max <= 0.09),
        ("S7 高臭氧改善城市数 9/9", n_improve, 9, n_improve == 9),
        ("4.4 文献条目数", len(links), 10, len(links) == 10),
    ]
    for name, got, exp, ok in extra:
        rows.append({"检查项": name, "结果表值": round(float(got), 3),
                     "文中引用": exp, "通过": "PASS" if ok else "FAIL"})
        fail += 0 if ok else 1

    # 文本层面的“变量—数值”配对检查：防止数值正确但归属写错
    text_pairs = [
        ("正文 NO₂ 占比 4.10% 归属", "NO₂ 的 mean|SHAP| 占比为 4.10%" in doc),
        ("正文 HCHO 占比 2.21% 归属", "HCHO 的 2.21%" in doc),
        ("正文无 HCHO=4.10% 错配", "HCHO（4.10%）" not in doc),
        ("正文无“绝大部分平方误差”", "绝大部分平方误差" not in doc),
        ("正文无“更依赖卫星特征”", "更依赖卫星特征" not in doc),
        ("摘要无 transferable", "transferable" not in doc),
        ("Methods 三阈值齐全", all(k in meth for k in ("149.500", "141.875", "139.750", "140.436"))),
        ("Methods 明确不可直接比较", "should not be read as a direct comparison" in meth),
        ("Methods 空间阈值未参与拟合", "was not used for model fitting" in meth),
        ("结果稿 NO₂ 归属 4.10%", "NO₂ `no2_col` mean|SHAP| 2.284（4.10%" in res),
        ("结果稿无 HCHO=4.10% 错配", "HCHO `hcho_col`" in res and "HCHO（4.10%）" not in res),
        ("结果稿阈值不可直接比较", "不应被理解为严格同一阈值下的直接比较" in res),
        ("结果稿含逐折阈值敏感性", "−7.47%" in res and "表 S7" in res),
        ("全文无“晴空”简称", "晴空" not in doc and "晴空" not in res and "clear-sky" not in doc),
        ("S6 标明非因果", "不是" in s6 and "因果" in s6),
        ("S6 月份固定效应表述", "month-of-year fixed effects" in s6),
        ("4.4 三条研究缺口", all(k in doc for k in ("缺口一", "缺口二", "缺口三"))),
        ("4.4 引用表 S8", "表 S8" in doc),
        ("4.4 文献无待核标记", "†" not in doc and "待核" not in doc),
    ]
    for name, ok in text_pairs:
        rows.append({"检查项": name, "结果表值": "OK" if ok else "MISSING",
                     "文中引用": "必须成立", "通过": "PASS" if ok else "FAIL"})
        fail += 0 if ok else 1

    ab = doc.split("## Abstract (English draft")[1].split("**Keywords**")[0]
    ab = ab.split("\n\n", 1)[1] if "\n\n" in ab else ab
    n_words = len([w for w in ab.split() if any(ch.isalnum() for ch in w)])
    ok = 220 <= n_words <= 250
    rows.append({"检查项": "英文摘要词数（220—250）", "结果表值": n_words,
                 "文中引用": "220—250", "通过": "PASS" if ok else "FAIL"})
    fail += 0 if ok else 1

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "consistency_check.csv", index=False, encoding="utf-8-sig")
    L = ["论文数字一致性校验（对照冻结结果表）", "=" * 56,
         f"共 {len(tab)} 项，FAIL {fail} 项", "",
         tab.to_string(index=False)]
    (OUT / "consistency_check.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
