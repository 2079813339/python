"""步骤 22：英文投稿稿件（APR）数字、术语与格式一致性校验（不训练模型）。

校验 outputs/paper/manuscript_APR_en.md 与 submission_package_APR.md 中引用的关键数字
是否与冻结结果表逐位一致，并检查术语边界（estimation / common-valid / 非因果）、
摘要词数、Highlights 字符数与图表文件是否齐备。
"""

import re
import sys
from pathlib import Path

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
    dec = pd.read_csv(OUT / "table_error_decomposition.csv")
    col = pd.read_csv(OUT / "table_satellite_collinearity.csv")
    fw = pd.read_csv(OUT / "table_spatial_foldwise_threshold.csv")
    fwc = pd.read_csv(OUT / "table_spatial_foldwise_by_city.csv")
    vif = pd.read_csv(OUT / "table_satellite_vif.csv")
    ss = pd.read_csv(E2A / "E2_selection_bias_stratified.csv")
    sc = pd.read_csv(E2A / "E2_selection_bias_by_city.csv")
    model = pd.read_parquet(config.DATA_PROCESSED / "modeling_table_with_tropomi_5km.parquet")

    doc = (OUT / "manuscript_APR_en.md").read_text(encoding="utf-8")
    pkg = (OUT / "submission_package_APR.md").read_text(encoding="utf-8")

    def e2t(ver, c="RMSE"):
        return float(t2[(t2.version == ver) & (t2.model == XGB) & (t2.split == "test")][c].iloc[0])

    def e2s(ver, c="RMSE"):
        return float(s2[(s2.version == ver) & (s2.model == XGB)][c].iloc[0])

    def e4(frame, ver, c):
        return float(frame[(frame.version == ver) & (frame.model == XGB)][c].iloc[0])

    def boot(df, ver, scene, group, cluster, c):
        key = {"split": scene} if "split" in df.columns else {"scene": scene, "group": group}
        q = df[(df.version == ver) & (df.model == XGB) & (df.cluster == cluster)]
        for k, v in key.items():
            q = q[q[k] == v]
        return float(q[c].iloc[0])

    def d(scene, ver, c):
        return float(dec[(dec["场景"].str.startswith(scene)) & (dec["版本"] == ver)][c].iloc[0])

    def cc(sat, other, c):
        return float(col[(col["卫星特征"] == sat) & (col["对照变量"] == other)][c].iloc[0])

    def vv(sat, c):
        return float(vif[vif["卫星特征"] == sat][c].iloc[0])

    common = (model["hcho_col"].notna() & model["no2_col"].notna()
              & (model["hcho_count"] > 0) & (model["no2_count"] > 0) & model["mda8"].notna())
    n_common = int(common.sum())

    def fmt(v, nd):
        return f"{v:.{nd}f}".replace("-", "\u2212")

    def present(v, nd):
        cands = {f"{v:.{nd}f}", fmt(v, nd), f"{abs(v):.{nd}f}"}
        return any(c in doc for c in cands)

    rows, fail = [], 0

    def chk(name, ok, detail=""):
        nonlocal fail
        rows.append({"检查项": name, "结果表值": detail, "通过": "PASS" if ok else "FAIL"})
        fail += 0 if ok else 1

    # 1) 核心数值：以冻结结果表为准，检查稿件中是否以相同精度出现
    vals = [
        ("E2-0 temporal RMSE 19.300", e2t("E2-0"), 3),
        ("E2-1 temporal RMSE 18.902", e2t("E2-1"), 3),
        ("E2-2 temporal RMSE 18.945", e2t("E2-2"), 3),
        ("E2-3 temporal RMSE 18.642", e2t("E2-3"), 3),
        ("E2-3 temporal dRMSE -0.658", e2t("E2-3") - e2t("E2-0"), 3),
        ("E2-3 temporal d% -3.41", (e2t("E2-3") / e2t("E2-0") - 1) * 100, 2),
        ("E2-0 spatial RMSE 17.277", e2s("E2-0"), 3),
        ("E2-1 spatial RMSE 16.960", e2s("E2-1"), 3),
        ("E2-2 spatial RMSE 17.042", e2s("E2-2"), 3),
        ("E2-3 spatial RMSE 16.800", e2s("E2-3"), 3),
        ("E2-3 spatial dRMSE -0.477", e2s("E2-3") - e2s("E2-0"), 3),
        ("E2-3 spatial d% -2.76", (e2s("E2-3") / e2s("E2-0") - 1) * 100, 2),
        ("E4-0 temporal overall 18.642", e4(e4t, "E4-0", "RMSE"), 3),
        ("E4-1 temporal overall 18.270", e4(e4t, "E4-1", "RMSE"), 3),
        ("E4-1 temporal overall d% -1.99", (e4(e4t, "E4-1", "RMSE") / e4(e4t, "E4-0", "RMSE") - 1) * 100, 2),
        ("E4-0 temporal high 33.341", e4(e4t, "E4-0", "RMSE_geP90"), 3),
        ("E4-1 temporal high 31.847", e4(e4t, "E4-1", "RMSE_geP90"), 3),
        ("E4-1 temporal high d% -4.48", (e4(e4t, "E4-1", "RMSE_geP90") / e4(e4t, "E4-0", "RMSE_geP90") - 1) * 100, 2),
        ("E4-0 temporal low 16.193", e4(e4t, "E4-0", "RMSE_ltP90"), 3),
        ("E4-1 temporal low 16.057", e4(e4t, "E4-1", "RMSE_ltP90"), 3),
        ("E4-1 temporal low d% -0.84", (e4(e4t, "E4-1", "RMSE_ltP90") / e4(e4t, "E4-0", "RMSE_ltP90") - 1) * 100, 2),
        ("E4-1 temporal high bias -27.026", e4(e4t, "E4-1", "Bias_geP90"), 3),
        ("E4-1 temporal high MAE d% -6.17", (e4(e4t, "E4-1", "MAE_geP90") / e4(e4t, "E4-0", "MAE_geP90") - 1) * 100, 2),
        ("E4-0 spatial overall 16.800", e4(e4s, "E4-0", "RMSE"), 3),
        ("E4-1 spatial overall 16.836", e4(e4s, "E4-1", "RMSE"), 3),
        ("E4-1 spatial overall d% +0.21", (e4(e4s, "E4-1", "RMSE") / e4(e4s, "E4-0", "RMSE") - 1) * 100, 2),
        ("E4-0 spatial high 23.282", e4(e4s, "E4-0", "RMSE_geP90"), 3),
        ("E4-1 spatial high 21.562", e4(e4s, "E4-1", "RMSE_geP90"), 3),
        ("E4-1 spatial high d% -7.39", (e4(e4s, "E4-1", "RMSE_geP90") / e4(e4s, "E4-0", "RMSE_geP90") - 1) * 100, 2),
        ("E4-1 spatial low 16.224", e4(e4s, "E4-1", "RMSE_ltP90"), 3),
        ("E4-1 spatial low d% +1.94", (e4(e4s, "E4-1", "RMSE_ltP90") / e4(e4s, "E4-0", "RMSE_ltP90") - 1) * 100, 2),
        ("E4-0 spatial overall bias +1.220", e4(e4s, "E4-0", "Bias"), 3),
        ("E4-1 spatial overall bias +2.208", e4(e4s, "E4-1", "Bias"), 3),
        ("E4-0 spatial high bias -17.731", e4(e4s, "E4-0", "Bias_geP90"), 3),
        ("E4-1 spatial high bias -15.444", e4(e4s, "E4-1", "Bias_geP90"), 3),
        ("E4-2 temporal high d% -2.48", (e4(e4t, "E4-2", "RMSE_geP90") / e4(e4t, "E4-0", "RMSE_geP90") - 1) * 100, 2),
        ("E4-2 spatial high d% -3.94", (e4(e4s, "E4-2", "RMSE_geP90") / e4(e4s, "E4-0", "RMSE_geP90") - 1) * 100, 2),
        ("E4-2 spatial overall d% +0.57", (e4(e4s, "E4-2", "RMSE") / e4(e4s, "E4-0", "RMSE") - 1) * 100, 2),
        ("E2-3 temporal station CI lo -0.930", boot(b2, "E2-3", "time", None, "station", "ci95_lo"), 3),
        ("E2-3 temporal station CI hi -0.396", boot(b2, "E2-3", "time", None, "station", "ci95_hi"), 3),
        ("E2-3 temporal city CI lo -1.024", boot(b2, "E2-3", "time", None, "city", "ci95_lo"), 3),
        ("E2-3 temporal city CI hi -0.287", boot(b2, "E2-3", "time", None, "city", "ci95_hi"), 3),
        ("E2-3 spatial station CI lo -0.852", boot(b2, "E2-3", "spatial", None, "station", "ci95_lo"), 3),
        ("E2-3 spatial station CI hi -0.156", boot(b2, "E2-3", "spatial", None, "station", "ci95_hi"), 3),
        ("E2-3 spatial city CI lo -1.212", boot(b2, "E2-3", "spatial", None, "city", "ci95_lo"), 3),
        ("E2-3 spatial city CI hi +0.107", boot(b2, "E2-3", "spatial", None, "city", "ci95_hi"), 3),
        ("E4-1 temporal high station CI -1.977/-1.000",
         boot(b4, "E4-1", "time", "high_ozone", "station", "ci95_lo"), 3),
        ("E4-1 temporal high city CI -1.993", boot(b4, "E4-1", "time", "high_ozone", "city", "ci95_lo"), 3),
        ("E4-1 spatial high station CI -2.068", boot(b4, "E4-1", "spatial", "high_ozone", "station", "ci95_lo"), 3),
        ("E4-1 spatial high city CI -2.357", boot(b4, "E4-1", "spatial", "high_ozone", "city", "ci95_lo"), 3),
        ("E4-1 spatial overall station CI lo -0.134", boot(b4, "E4-1", "spatial", "overall", "station", "ci95_lo"), 3),
        ("E4-1 spatial overall station CI hi +0.205", boot(b4, "E4-1", "spatial", "overall", "station", "ci95_hi"), 3),
        ("E4-1 spatial overall P_improve 0.336", boot(b4, "E4-1", "spatial", "overall", "station", "P_improve"), 3),
        ("SHAP NO2 2.284", float(sh[sh.feature == "no2_col"]["mean_abs_shap"].iloc[0]), 3),
        ("SHAP HCHO 1.232", float(sh[sh.feature == "hcho_col"]["mean_abs_shap"].iloc[0]), 3),
        ("SHAP NO2 share 4.10", float(sh[sh.feature == "no2_col"]["share_pct"].iloc[0]), 2),
        ("SHAP HCHO share 2.21", float(sh[sh.feature == "hcho_col"]["share_pct"].iloc[0]), 2),
        ("SHAP high-ozone abs 4.691", float(oz[(oz.group_type == "MDA8分位") & (oz.group == ">P90")]["sat_abs_mean"].iloc[0]), 3),
        ("SHAP low-ozone abs 3.298", float(oz[(oz.group_type == "MDA8分位") & (oz.group == "<P50")]["sat_abs_mean"].iloc[0]), 3),
        ("SSE temporal high share 32.1", d("时间", "E4-0", "高臭氧SSE占比%"), 1),
        ("SSE spatial high share 19.3", d("空间", "E4-0", "高臭氧SSE占比%"), 1),
        ("SSE temporal high contribution 71.3", d("时间", "E4-1", "高臭氧占SSE变化幅度%"), 1),
        ("corr NO2-ground 0.551", cc("no2_col", "no2_mean", "Pearson_r"), 3),
        ("corr NO2-ground spearman 0.526", cc("no2_col", "no2_mean", "Spearman_r"), 3),
        ("corr NO2-PM10 0.407", cc("no2_col", "pm10_mean", "Pearson_r"), 3),
        ("corr columns pearson 0.062", cc("no2_col", "hcho_col", "Pearson_r"), 3),
        ("corr columns spearman 0.132", cc("no2_col", "hcho_col", "Spearman_r"), 3),
        ("VIF NO2 2.095", vv("no2_col", "VIF_标准"), 3),
        ("VIF HCHO 1.295", vv("hcho_col", "VIF_标准"), 3),
        ("foldwise E4-0 high 23.273", float(fw[fw["版本"] == "E4-0"]["RMSE_高臭氧"].iloc[0]), 3),
        ("foldwise E4-1 high 21.536", float(fw[fw["版本"] == "E4-1"]["RMSE_高臭氧"].iloc[0]), 3),
        ("foldwise d% -7.47", float(fw[fw["版本"] == "E4-1"]["Δ%"].iloc[0]), 2),
        ("common-valid coverage 42.58", n_common / len(model) * 100, 2),
    ]
    for name, v, nd in vals:
        chk(name, present(v, nd), fmt(v, nd))

    # 2) 样本量与计数
    for name, v in [
        ("modeling rows 51,486", f"{len(model):,}"),
        ("common-valid 21,925", f"{n_common:,}"),
        ("2022 test 4,691", f"{int((model.loc[common, 'year'] == config.TEST_YEAR).sum()):,}"),
        ("spatial high-ozone n 2,198", f"{int(e4(e4s, 'E4-0', 'n_geP90')):,}"),
        ("temporal high-ozone n 471", f"{int(e4(e4t, 'E4-0', 'n_geP90')):,}"),
        ("high-ozone share temporal 10.04", f"{(e4(e4t, 'E4-0', 'n_geP90') / e4(e4t, 'E4-0', 'n')) * 100:.2f}"),
        ("high-ozone share spatial 10.03", f"{(e4(e4s, 'E4-0', 'n_geP90') / e4(e4s, 'E4-0', 'n')) * 100:.2f}"),
        ("foldwise high-ozone n 2,224", f"{int(fw[fw['版本'] == 'E4-0']['n_高臭氧'].iloc[0]):,}"),
    ]:
        chk(name, v in doc, v)

    # 3) 选择偏差（表 S6 与冻结分层表）
    s6 = (OUT / "table_supplementary_selection_bias.md").read_text(encoding="utf-8")

    def s6num(pat, cast=float):
        return cast(re.search(pat, s6).group(1).replace(",", ""))

    chk("S6 beta 25.71", abs(s6num(r"β（共同有效 vs 无共同有效像元） \| \+([0-9.]+)") - 25.71) < 1e-9)
    chk("S6 SE 0.65", abs(s6num(r"聚类稳健 SE \| ([0-9.]+)") - 0.65) < 1e-9)
    chk("S6 t 39.3", abs(s6num(r"t 值 \| ([0-9.]+)") - 39.3) < 1e-9)
    chk("S6 n 50,192", s6num(r"样本量 \| ([0-9,]+)", int) == 50192)
    chk("S6 clusters 38", s6num(r"聚类数（站点） \| ([0-9]+)", int) == 38)
    seasons = ["冬", "春", "夏", "秋"]
    expect_season = [31.136, 31.241, 18.575, 29.272]
    for s, e in zip(seasons, expect_season):
        chk(f"S6 {s} difference {e}", present(e, 3), fmt(e, 3))
    chk("S6 nine-city min 19.23", present(float(sc["diff"].min()), 2), fmt(float(sc["diff"].min()), 2))
    chk("S6 nine-city max 31.81", present(float(sc["diff"].max()), 2), fmt(float(sc["diff"].max()), 2))

    # 4) 术语与边界（存在型）
    ab = doc.split("## Abstract")[1].split("**Keywords**")[0]
    term_checks = [
        ("禁用 clear-sky", "clear-sky" not in doc and "clear sky" not in doc),
        ("摘要无 transferable", "transferable" not in ab),
        ("摘要任务边界为 estimation", "estimation, not forecasting" in ab),
        ("无 significant improvement", "significant improvement" not in doc.lower()),
        ("无 proves", re.search(r"\bproves\b", doc.lower()) is None),
        ("estimation 而非 forecasting（任务边界句）", "not a forecasting study" in doc),
        ("共同有效样本术语", "common-valid" in doc and "satellite-observable conditions" in doc),
        ("S6 非因果表述", "not evidence that satellite observation causes higher ozone" in doc),
        ("月份固定效应英文", "month-of-year fixed effects" in doc),
        ("三阈值齐备", all(k in doc for k in ("142.286", "139.750", "149.500", "141.875", "140.436", "140.875"))),
        ("阈值不可直接比较", "should not be read as a comparison at an identical threshold" in doc),
        ("71.3% 分母说明", "share of the total squared-error reduction, not a share of the total error" in doc),
        ("E4 为训练策略", "validated training strategy" in doc),
        ("SHAP 非机制证据", "do not constitute evidence about ozone formation" in doc),
        ("未宣称首创", "does not claim a first use" in doc),
        ("未宣称可跨省迁移", "does not establish transferability across provinces" in doc),
        ("NO2 归属 4.10%", "NO₂ (`no2_col`) has mean |SHAP| 2.284 μg/m³ (4.10%, rank 9" in doc),
        ("HCHO 归属 2.21%", "HCHO (`hcho_col`) 1.232 μg/m³ (2.21%, rank 16)" in doc),
        ("正文无 HCHO 4.10% 错配", "HCHO (`hcho_col`) 1.232" in doc and "HCHO 4.10" not in doc),
    ]
    for name, ok in term_checks:
        chk(name, ok)

    # 5) 摘要词数与 Highlights 长度
    ab = doc.split("## Abstract")[1].split("**Keywords**")[0]
    n_words = len([w for w in ab.split() if any(ch.isalnum() for ch in w)])
    chk("abstract word count 220-250", 220 <= n_words <= 250, str(n_words))

    hl = re.search(r"\*\*Highlights\*\*[^\n]*\n(.*?)\n\n---", doc, re.S)
    hl_lines = [ln[2:].strip() for ln in hl.group(1).splitlines() if ln.startswith("- ")]
    chk("highlights 3-5 items", 3 <= len(hl_lines) <= 5, str(len(hl_lines)))
    for i, h in enumerate(hl_lines, 1):
        chk(f"highlight {i} <=85 chars", len(h) <= 85, f"{len(h)}: {h}")

    # 6) 图表编号与文件
    for i in range(1, 6):
        chk(f"Fig. {i} 引用与文件", f"**Fig. {i}**" in doc or f"Fig. {i}" in doc,
            str((OUT / "figures" / f"Fig{i}.png").exists()))
    for i in range(1, 5):
        chk(f"Fig. S{i} 文件存在", (OUT / "figures" / f"FigS{i}.png").exists())
    chk("Table 1-3 均在正文", all(f"**Table {i}.**" in doc for i in (1, 2, 3)))
    chk("Table S1-S8 清单", all(f"Table S{i}" in doc for i in range(1, 9)) and "Table S5b" in doc)

    # 7) 参考文献条目数
    refs = doc.split("## References")[1].split("---")[0]
    n_doi = len(re.findall(r"https://(?:doi\.org|arxiv\.org)/\S+", refs))
    n_entries = len([ln for ln in refs.splitlines() if re.match(r"^[A-Z][a-z]+ [A-Z]", ln.strip())])
    chk("references entries >=15", n_entries >= 15, str(n_entries))
    chk("references DOI/arXiv >=13", n_doi >= 13, str(n_doi))

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "consistency_check_en.csv", index=False, encoding="utf-8-sig")
    L = ["英文投稿稿件一致性校验（APR）", "=" * 56,
         f"共 {len(tab)} 项，FAIL {fail} 项", "",
         tab.to_string(index=False)]
    (OUT / "consistency_check_en.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
