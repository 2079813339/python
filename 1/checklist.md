# 论文整合收口清单

更新时间：2026-11-16。状态说明：✅ 已完成 / ◻ 待办。

| # | 项目 | 状态 | 证据 |
|---|---|---|---|
| 1 | 确认空间高臭氧评价 P90 定义并写入方法部分 | ✅ | `outputs/e4/audit/e4_high_ozone_group_definition.txt`、`e4_per_fold_threshold.csv`；结论：训练阈值逐折独立（140.44—143.13），评价阈值统一为 141.875（n=2,198） |
| 2 | 将 E2、SHAP、E4 主结果合并为统一结果表 | ✅ | `outputs/paper/table_main_results.csv/.md` |
| 3 | 统一 RMSE/MAE/Bias/R² 的单位、精度与正负号定义 | ✅ | `results_section_draft.md` 第 0 节（Δ=新−旧，Bias=预测−观测） |
| 4 | 时间与空间 Bootstrap 结果整理为一张表 | ✅ | `outputs/paper/table_bootstrap.csv/.md`（12 行，含 P_improve 与簇数） |
| 5 | 整理 3/5/10 km 缓冲区敏感性 | ✅ | `table_supplementary_sensitivity.*` + `outputs/tables/e2_buffer_sensitivity.csv` |
| 6 | 整理有效像元覆盖率与共同有效样本选择偏差 | ✅ | `table_coverage.*`、`table_supplementary_sensitivity.*`、`outputs/reports/e2_robustness_report.md` |
| 7 | 核对图表、样本量、模型版本与冻结哈希 | ✅ | `table_source_audit.csv`（14 个源文件 SHA256 前缀）；冻结清单 `outputs/e2/frozen/E2_SHA256SUMS.txt`（56 项）与 `E2_SHA256SUMS_shap_archive.txt` 均未改动 |
| 8 | 撰写讨论、局限性与结论 | ✅ | `discussion_limitations_conclusion_abstract.md`（4.1—4.5、Limitations 6 项、Conclusion 4 条；4.4 已按三条研究缺口重写，对照矩阵移至表 S8） |

## 写入阶段（阶段 1—5）执行清单

| # | 项目 | 状态 | 证据 |
|---|---|---|---|
| W1 | Discussion 初稿（4.1—4.5） | ✅ | `discussion_limitations_conclusion_abstract.md`；4.4 已含 10 条可核对对照文献与协议/口径说明，未虚构引用 |
| W2 | Limitations（6 项） | ✅ | 同上第 5 节，逐项附量化依据（42.58%、β≈+25.7、108—454 行等） |
| W3 | Conclusion（4 条） | ✅ | 同上第 6 节，未超出时间/空间验证支持范围 |
| W4 | 英文 Abstract（238 词） | ✅ | 同上；含 E2 空间未达门槛、E4 低臭氧代价，未只选改善指标；“transferable”已改为“consistent across the temporal and within-region spatial validation protocols” |
| W5 | 全文一致性检查（数值 + 归属） | ✅ | `consistency_check.txt`：**109 项**对照冻结表，FAIL=0（`scripts/21_check_manuscript_numbers.py`）；含 S5 VIF/相关（含两柱互相关）、S6 β与四季、S7 逐折阈值、文献条目数与“变量—数值配对”文本检查 |
| W6 | 目标期刊筛选与投稿准备 | ✅ | `w6_journal_targets.md`：8 本候选 + 3 档定位（**主投 APR** / 冲刺 EP / 备选 AE / 保底 Remote Sensing），含范围关键句、近期同型论文（APR 4 篇 2024—2026）、APC 与周期口径、文章限制；JCR 与中科院分区须分别核实 |
| W7 | APR 格式英文全文整合 | ✅ | `manuscript_APR_en.md`：Title→Abstract(238 词)/Keywords(6)→1 Introduction（三条缺口+三贡献）→2 Methods（含 E2 析因、SHAP、E4 三阈值、嵌套 LOCO、选择偏差回归）→3 Results→4 Discussion→5 Limitations→6 Conclusions→Data/Code Availability→CRediT→Competing Interest→Acknowledgements→References(15) |
| W8 | 投稿文件包与编号统一 | ✅ | `submission_package_APR.md`（metadata、Highlights 5 条均 ≤85 字符、Cover Letter、声明、文件清单、待作者补充项）；`outputs/paper/figures/Fig1—Fig5.png`、`FigS1—FigS4.png` 已完成重命名；Table 1—3 与 Table S1—S8（含 S5b）编号在正文与清单中一致 |
| W9 | 英稿件数字与术语一致性检查 | ✅ | `consistency_check_en.txt`：**131 项 PASS、FAIL=0**（`scripts/22_check_manuscript_english.py`）；核对 E2/E4/SHAP/SSE/相关/VIF/逐折阈值/选择偏差等冻结数值，并检查 clear-sky 禁用、非因果表述、71.3% 分母说明、摘要词数与 Highlights 长度 |

## 审稿前修订清单（R1—R7）

| # | 项目 | 状态 | 证据 |
|---|---|---|---|
| R1 | 修正 4.1 HCHO/NO₂ SHAP 占比对应关系 | ✅ | 改为“NO₂ 4.10% 高于 HCHO 2.21%”；`consistency_check.csv` 增加正向归属与反向错配两项文本检查 |
| R2 | 4.3 高臭氧组平方误差贡献重新核算 | ✅ | `outputs/paper/table_error_decomposition.md`（表 S4）：时间高臭氧 SSE 占比 32.1%（非“绝大部分”）、贡献 71.3% 的 SSE 减少量；空间 19.3% |
| R3 | Methods 补充 E4 三类阈值定义 | ✅ | `outputs/paper/methods_e4_threshold_insert.md`（中英对照 + 阈值一览表） |
| R4 | 卫星特征与地面变量共线性核查 | ✅ | `table_satellite_collinearity.md`（表 S5）：NO₂ 柱 vs 地面 NO₂ r=0.551（Spearman 0.526），vs PM₁₀ r=0.407，HCHO 与地面污染物 \|r\|≤0.09；`table_satellite_vif.md`（表 S5b）为标准 VIF（辅助回归含另一卫星特征）NO₂ 2.095、HCHO 1.295，纳入另一卫星特征后三位小数无变化；两柱互相关 Pearson r=0.062、Spearman ρ=0.132（n=21,925），4.1 已改为先报 VIF 变化、再报互相关 |
| R5 | 收紧 4.2 依赖/单调表述与摘要迁移表述 | ✅ | 4.2 改为“mean\|SHAP\| 较高、平均贡献幅度更大”；HCHO 限定为“当前模型与样本范围内总体单调递增” |
| R6 | 4.4 文献查新与对照矩阵 | ✅ | 已按三条研究缺口重写，矩阵移至 `table_literature_matrix.md`（表 S8，10 条，含 DOI/年份/原文指标）；10 条题名/期刊/DOI 核对完毕，6 条带数值者已逐条核对原文摘要或正文（含本轮补核 Chen J. 2022：RMSE 10.25/MAE 7.30/R² 0.912、日均值口径；Wang Y. 2021：O₃ R² 0.91、样本级 CV） |
| R7 | 数值一致性检查升级为“变量—数值配对” | ✅ | `consistency_check.txt` **109 项 PASS**；含 S5/S6/S7 数值、两柱互相关、文献条目数、HCHO/NO₂ 归属正反两向、禁用表述、Methods 三阈值、两侧阈值不可直接比较、摘要 238 词 |

术语统一（estimation 而非 forecasting；MDA8 ozone；nested leave-one-city-out validation）已在
`discussion_limitations_conclusion_abstract.md` 开头固定，摘要与结论均遵守。

## 图表清单（供正文与补充材料引用）

| 图 | 文件 |
|---|---|
| Fig. 1 全局 SHAP 贡献 | `outputs/paper/figures/Fig1.png`（原名 `fig1_global.png`） |
| Fig. 2 HCHO/NO₂ 分位依赖 | `outputs/paper/figures/Fig2.png`（原名 `fig2_dependence.png`） |
| Fig. 3 E4 总体性能 | `outputs/paper/figures/Fig3.png`（原名 `Fig11_e4_overall_performance.png`） |
| Fig. 4 E4 高臭氧性能 | `outputs/paper/figures/Fig4.png`（原名 `Fig12_e4_high_ozone_performance.png`） |
| Fig. 5 城市异质性 | `outputs/paper/figures/Fig5.png`（原名 `Fig13_e4_city_heterogeneity.png`） |
| Fig. S1 HCHO×NO₂ 交互 | `outputs/paper/figures/FigS1.png`（原名 `fig3_interaction.png`） |
| Fig. S2 城市卫星占比 vs ΔRMSE | `outputs/paper/figures/FigS2.png`（原名 `fig4_city.png`） |
| Fig. S3 分组/季节卫星贡献 | `outputs/paper/figures/FigS3.png`（原名 `fig5_ozone.png`） |
| Fig. S4 E4 误差分布 | `outputs/paper/figures/FigS4.png`（原名 `Fig14_e4_error_distribution.png`） |
| 表 S1 敏感性（3/5/10 km、城市异质、选择偏差） | `outputs/paper/table_supplementary_sensitivity.md` |
| 表 S2 SHAP 贡献 | `outputs/paper/table_supplementary_shap.md` |
| 表 S3 覆盖率 | `outputs/paper/table_coverage.md` |
| 表 S4 E4 平方误差分解 | `outputs/paper/table_error_decomposition.md` |
| 表 S5 卫星—地面共线性 | `outputs/paper/table_satellite_collinearity.md` |
| 表 S5b 卫星特征 VIF（标准口径 + F1 辅助诊断） | `outputs/paper/table_satellite_vif.md` |
| 表 S6 选择偏差回归（共同有效样本） | `outputs/paper/table_supplementary_selection_bias.md` |
| 表 S7 逐折阈值敏感性 | `outputs/paper/table_spatial_foldwise_threshold.md` |
| 表 S8 文献对照矩阵（任务/输入/协议/原文指标） | `outputs/paper/table_literature_matrix.md` |

## 冻结状态

- E2（12/18 号）与 E4（19 号）结果均已冻结；19 号内含 E4-0 与冻结 E2-3 的逐位一致性断言。
- 18b 验收记录：`outputs/e2/audit/E2_shap_alignment_acceptance.txt`（两次独立重建逐行差 = 0，九折指标相对差 = 0）。
- 后续不再新增 E3、CatBoost、新权重档位或模型调参。
