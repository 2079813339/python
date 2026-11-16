# Satellite precursor columns in regional near-surface ozone estimation: conditional incremental value, satellite-sample representativeness, and high-ozone reweighting in Fujian Province, China

**Article type**: Original research article
**Target journal**: Atmospheric Pollution Research (Elsevier)

**Authors**: [Author 1]¹, [Author 2]¹, [Author 3]², [Corresponding author]¹\*
**Affiliations**: ¹[Department, Institution, City, Country]; ²[Department, Institution, City, Country]
\*Corresponding author: [name], [email], [postal address]

---

## Abstract

Near-surface ozone estimation in Fujian Province, southeast China, relies mainly on ground observations and meteorology; whether satellite precursor columns add usable information, and whether that information can be converted into better high-ozone estimation, remains unclear. Using daily maximum 8-h average (MDA8) ozone at 38 CNEMC stations (2019–2022), TROPOMI HCHO and NO₂ columns aggregated to 5 km buffers, ERA5 meteorology and ground pollutants, we trained XGBoost under a temporally independent protocol (2019–2020 training, 2021 validation, 2019–2021 refit, 2022 test) and a nested leave-one-city-out protocol. The F1 baseline reached RMSE 19.300 μg/m³ (temporal) and 17.277 μg/m³ (spatial). Adding HCHO and NO₂ reduced temporal RMSE to 18.642 μg/m³ (−3.41%) and spatial RMSE to 16.800 μg/m³ (−2.76%); the latter did not meet the pre-specified −0.5 μg/m³ criterion, and its city-cluster bootstrap interval crossed zero. SHAP attributed 6.32% of total contribution to the two satellite features, with larger contributions in high-ozone and summer samples. Reweighting training samples above the training-period 90th percentile (λ=2) reduced high-ozone RMSE by 4.48% (temporal, 33.341→31.847 μg/m³) and 7.39% (spatial, 23.282→21.562 μg/m³), while spatial overall RMSE changed by +0.21% (bootstrap interval crossing zero) and low-ozone RMSE increased by 1.94%. Satellite precursor information therefore provides conditional incremental value, and high-ozone-weighted training yields a targeted improvement that is consistent across the temporal and within-region spatial validation protocols, rather than a general enhancement across all concentration ranges. All results pertain to satellite-observable (common-valid) conditions (42.58% sample coverage) and describe estimation, not forecasting.

**Keywords**: near-surface ozone; TROPOMI; XGBoost; high-ozone reweighting; nested leave-one-city-out validation; Fujian Province

**Highlights** (≤85 characters each; optional for this journal)
- Same-sample ablation isolates the incremental value of HCHO and NO₂ columns.
- Satellite gains are clear in time but weak and city-dependent in space.
- Common-valid samples are systematically biased toward higher ozone (+25.71 μg/m³).
- High-ozone reweighting cuts high-ozone RMSE by 4.48–7.39%.
- High-ozone gains come with a +0.21% overall RMSE change in spatial validation.

---

## 1. Introduction

Near-surface ozone is a secondary pollutant whose daily maximum 8-h average (MDA8) is the quantity most directly linked to human exposure and to regulatory reporting. In southeast China, ozone formation is controlled by a combination of strong solar radiation, high temperature, monsoon-driven transport and intense precursor emissions, and coastal and inland cities differ markedly in photochemical regime and boundary-layer structure (Wang Y. et al., 2021; Chen G. et al., 2021). Dense ground networks such as the China National Environmental Monitoring Centre (CNEMC) provide accurate point measurements, but they remain sparse relative to the spatial variability of ozone, and they do not observe precursors directly.

Satellite retrievals have therefore been used to enrich surface ozone estimation. Total column NO₂ and HCHO are the most frequently adopted precursor proxies, either as direct predictors of surface ozone (Chen J. et al., 2022; Chen X. et al., 2024) or as inputs to full-coverage mapping products (Wang Y. et al., 2021). Machine learning (ML) models, particularly tree ensembles, are now routinely used to fuse ground observations, reanalysis meteorology and satellite columns into high-resolution estimates (Chen G. et al., 2021; Chen J. et al., 2022). Three issues remain insufficiently resolved, and they structure this study.

**Gap 1 — the incremental value of satellite inputs is not homogeneous.** Regional products have advanced quickly: stacked machine-learning estimates across China (Li Z. et al., 2024), FY-4A/B satellite–machine-learning fusion for 24-h ozone (Wang Y. et al., 2025), reanalysis-based retrieval for pollution events (Wang L. et al., 2025) and TROPOMI-based analyses of spring-festival ozone enhancement (Zhu et al., 2024) share the premise that adding remote-sensing or fused predictors improves the estimate. The size of that improvement, however, is not transferable. Studies that add satellite columns report improvements that differ in magnitude and location: Wang W. et al. (2022) found that TROPOMI data mainly improved extreme-value estimation in California, whereas Chen X. et al. (2024) reported that reconstructed columns mainly raised coverage, with accuracy close to a satellite-free configuration. Because most studies change several components at once (sample, predictors, validation split), a reported gain cannot be transplanted as a prior expectation. What is missing is a controlled ablation in which the same rows, the same split and the same model are used, so that the difference between versions can be attributed to the satellite inputs rather than to sample composition.

**Gap 2 — random and spatial validation answer different questions.** Random or sample-level cross-validation mainly measures interpolation within a population, whereas spatial hold-out measures extrapolation to locations that did not contribute to training. Seamless mapping frameworks (Li T. et al., 2025) and hourly geostationary retrievals (Li S. et al., 2025) reduce the missing-pixel problem, but they change what is being validated, because gap-filled estimates must be judged under the same hold-out design as observed ones. Chen G. et al. (2021) reported R² 0.84 under sample-level CV but 0.79 under station-level CV for the same model, and Young et al. (2016) showed that conventional K-fold and leave-one-out schemes overestimate performance away from monitoring sites; Roberts et al. (2017) formalized blocked cross-validation for dependent data. An improvement measured under a random split therefore cannot be read as cross-city generalization, and a regional study should test both.

**Gap 3 — overall accuracy is not the same objective as high-ozone accuracy.** Most estimation studies report a single overall RMSE or R², or an exceedance-classification score, and rarely report stratified metrics together with the overall cost. Deng T. et al. (2022) used a weighted loss to emphasize high-ozone errors, but the task was 1–3-day forecasting and the cost to the non-high-ozone range was not reported alongside the gain. A model with better overall performance does not necessarily provide adequate estimates under high-ozone conditions, and a targeted weighting strategy should be judged on both its gain and its cost. These concerns are emphasized in recent reviews of machine learning for tropospheric ozone (Hickman et al., 2025) and of overlooked issues in ML air-quality modelling, including sample imbalance and validation design (Tang et al., 2024).

A fourth issue is observational rather than methodological. Satellite columns are only available where retrieval-quality pixels exist, so the samples used to train and evaluate any satellite-enhanced model are a subset of the ground samples. If that subset is not representative, conclusions obtained under it cannot be extended to all weather conditions. This issue is usually acknowledged qualitatively but rarely quantified.

This study addresses these four issues for Fujian Province, southeast China, using 2019–2022 MDA8 ozone at CNEMC stations, TROPOMI HCHO and NO₂ columns, ERA5 meteorology and ground pollutant measurements. We pursue three objectives. The first is to isolate, on identical samples and splits, the incremental value of HCHO, of NO₂, and of both, and to test that value under a temporally independent protocol and a nested leave-one-city-out (LOCO) protocol; SHAP is used here only to describe how the trained model uses the satellite features, and the LOCO protocol is the spatial validation of this first objective rather than a separate contribution. The second is to quantify whether common-valid samples are representative of the remaining ground observations, which delimits the conditions under which satellite-enhanced estimates apply. The third is to test whether a high-ozone-weighted training objective improves high-ozone estimation, and at what cost to the rest of the concentration range.

The study is deliberately framed as an application-methods contribution. It does not propose a new learning algorithm, does not claim a first use of satellite data for ozone estimation, and does not infer photochemical regimes from model explanations: without VOC or PAN measurements, SHAP values are treated strictly as descriptions of model behaviour. All results refer to estimation (mapping) under satellite-observable conditions, not to forecasting.

## 2. Materials and methods

### 2.1 Study area and ground observations

Fujian Province (23°30′–28°22′N, 115°50′–120°40′E) comprises nine prefecture-level cities: Fuzhou, Xiamen, Quanzhou, Zhangzhou, Putian and Ningde along the coast, and Nanping, Sanming and Longyan inland. The province spans a humid subtropical monsoon climate with marked seasonal differences in radiation, rainfall and boundary-layer height.

Hourly surface ozone was obtained from the CNEMC hourly dataset for China (Zenodo record 10.5281/zenodo.10911197) for 2019–2022, in Beijing time (UTC+8). Of 41 candidate stations, two had no valid ozone record during the study period and were excluded; the modeling table contains 39 stations, of which 38 contributed common-valid satellite samples. Daily MDA8 was computed from 17 sliding 8-h windows (starting at 00–16 local time); a window was accepted with at least six valid hourly values, and a day was accepted with at least six valid windows. Daily means of NO₂, CO, PM₂.₅, PM₁₀ and SO₂ at the same stations were taken from the CNEMC real-time release archive (quotsoft historical air-quality dataset). The resulting modeling table contains 51,486 station-days with a valid MDA8 label.

### 2.2 Satellite precursor columns and the common-valid sample

Daily Level-3 TROPOMI HCHO and NO₂ products were obtained from the Google Earth Engine collections `COPERNICUS/S5P/OFFL/L3_HCHO` and `COPERNICUS/S5P/OFFL/L3_NO2`, using the tropospheric column number density band of each product. Pixels with `cloud_fraction` above 0.30 were discarded. Each Beijing-time (UTC+8) calendar day was matched to the overpass imagery of the previous 16:00 UTC to 16:00 UTC window, so that the daily station value refers to the same local day as the MDA8 label. HCHO and NO₂ were extracted together under a common valid-pixel mask: a station-date was retained only when the matched overpass provided at least one valid pixel for both products within the buffer. The buffer mean was computed from those co-valid pixels, and pixels from different overpasses of the same day were pooled before averaging. Buffers of 5 km (primary), 3 km and 10 km were evaluated. No interpolation, gap filling, zero substitution or negative-value truncation was applied, so missing values remained missing. The 0.01° grid is a product grid, not the native retrieval footprint (approximately 3.5 × 7 km²), so adjacent grid cells are not independent observations.

A station-day was defined as *common-valid* when both HCHO and NO₂ had at least one quality-screened valid pixel within the buffer and the MDA8 label was present. Common-valid station-days number 21,925, i.e. 42.58% of the modeling table (Table S3). Buffers of 3 km and 10 km yielded 20,481 and 24,467 common-valid station-days and were used only as sensitivity checks. The HCHO/NO₂ ratio was computed only when both columns were strictly positive (21,223 station-days, 4,534 test rows) and was used only in a supplementary experiment; it is a proxy for chemical regime, not a VOC/NOx sensitivity classification.

Common-valid status is not synonymous with cloud-free conditions. It depends jointly on cloud screening, retrieval quality, spatial coverage and the number of valid pixels, and it may also favour certain radiation and boundary-layer conditions. We therefore use the terms *common-valid observations* and *satellite-observable conditions* throughout.

### 2.3 Predictors

The F1 predictor set contains 28 features in four groups: (i) three geographic features (longitude, latitude, elevation); (ii) four calendar features (month, day of year, and its sine and cosine transforms); (iii) sixteen ERA5/ERA5-Land meteorological variables aggregated to daily values at each station (2 m mean, maximum and minimum temperature, dew-point temperature, relative humidity, surface pressure, precipitation, cloud cover, downward shortwave radiation, wind speed, u and v wind components, maximum and mean boundary-layer height, maximum vapour-pressure deficit, soil moisture); and (iv) five ground pollutant daily means (NO₂, CO, PM₂.₅, PM₁₀, SO₂). Meteorological series were obtained from the Open-Meteo Historical Weather API, which serves ERA5 and ERA5-Land reanalysis (Hersbach et al., 2020). The F0 set contains the first three groups only and serves as the no-pollutant reference. Satellite experiments add `hcho_col` and `no2_col` to F1.

All predictors are same-day values. The target is therefore estimated, not forecast: no lagged or future information is used, and the results carry no forecasting claim.

### 2.4 Models and the ablation design

Five regressors were trained on identical rows, features and splits: linear regression (B0), random forest (B1), XGBoost (B2), LightGBM (B3) and extremely randomized trees (B4). XGBoost was designated the primary model before the satellite experiments; the others are reported as algorithm controls. Hyperparameters were selected on the 2021 validation year using a shared grid (for XGBoost, maximum tree depth over {4, 6, 8}; for the forests, number of trees over {300, 500} and minimum leaf size over {1, 3, 5}; for LightGBM, number of leaves over {15, 31, 63}); the selected depth for F1 was 6. Predictors were median-imputed and, for the linear model, standardized within each training fold.

The satellite ablation (E2) keeps rows, splits and model settings fixed and changes only the predictor block: E2-0 = F1; E2-1 = F1 + HCHO; E2-2 = F1 + NO₂; E2-3 = F1 + HCHO + NO₂. E2-0 to E2-3 use the same 21,925 common-valid rows, so differences between versions cannot be attributed to sample composition. A fifth version (E2-4 = F1 + HCHO + NO₂ + ratio) is evaluated on the 21,223 positive-ratio subsample and is reported only as a supplementary experiment, because its sample differs by 702 rows and cannot be compared directly with E2-0 to E2-3.

The retention decision was fixed before the experiments: satellite features were to be retained only if RMSE decreased by at least 0.5 μg/m³ and at least 1% under *both* the temporal and the spatial protocols. This criterion is reported as a pre-specified study rule; the spatial outcome did not satisfy it, and we report that outcome as such rather than redefining the threshold afterwards.

### 2.5 Validation protocols

**Temporal protocol.** Models were fitted on 2019–2020, hyperparameters were selected on 2021 with the 2019–2020 fit, the final model was refitted on 2019–2021 with the selected settings, and performance was evaluated on the independent 2022 year (4,691 common-valid rows). The test year influenced neither fitting nor tuning.

**Nested leave-one-city-out protocol.** Nine outer folds were defined by holding out one city at a time. In each fold, the outer model was refitted on all rows of the eight training cities for 2019–2022 and evaluated on all rows of the held-out city. Hyperparameters were selected inside the outer training set only: the eight training cities were split into 2019–2020 (fit) and 2021 (validation), so the held-out city influenced neither the hyperparameters nor any threshold. Pooling the nine held-out predictions yields 21,925 rows. This protocol measures within-province cross-city applicability; it does not establish transferability across provinces, climate zones or satellite-observing conditions.

**Uncertainty.** Differences between paired models were resampled by cluster bootstrap, with stations (38 clusters) as the primary unit and cities (9 clusters) as a sensitivity unit; both models were evaluated on the same resampled clusters. We report percentile 95% intervals and the bootstrap probability of improvement, *P*<sub>improve</sub> (the fraction of replicates with a negative RMSE difference). *P*<sub>improve</sub> is a resampling probability, not a conventional p-value.

### 2.6 Model interpretation

The frozen E2-3 model was interpreted with TreeSHAP (Lundberg & Lee, 2017; Lundberg et al., 2020) on the 2022 test set (n = 4,691). Additivity was verified: the maximum absolute deviation between the sum of SHAP values plus the model bias and the model prediction was 2.75 × 10⁻⁴ μg/m³, arising from single-precision input. Contributions are summarized as mean absolute SHAP values, as signed contributions, by decile of the feature value, by season, by city and by ozone subgroup. Because Shapley values are not reliable feature-importance measures under correlated predictors (Kumar et al., 2020), SHAP results are interpreted only as descriptions of how the trained model distributes credit among features, not as evidence about atmospheric chemistry.

### 2.7 High-ozone reweighting

The reweighting experiment (E4) keeps predictors, rows, splits and the spatial protocol unchanged and modifies only the training objective. Training samples with MDA8 above a training-period 90th-percentile threshold received weight λ, and all other samples received weight 1; weights entered the model through the sample-weight interface, without row duplication. Three weight settings were pre-registered (λ = 1, 1.5 and 2); λ = 2 is the primary setting (E4-1) and λ = 1.5 is a sensitivity setting (E4-2). E4-0 is the unweighted model and is identical to the frozen E2-3 fit.

Three thresholds were used for different purposes and must not be conflated (Table 3). First, the *weighting threshold* was re-estimated from the training samples of each fit and never from held-out cities or evaluation years: 142.286 μg/m³ for 2019–2020 (hyperparameter selection) and 139.750 μg/m³ for the 2019–2021 refit in the temporal protocol; in the spatial protocol it was re-estimated independently within the eight training cities of each outer fold (140.436–143.125 μg/m³, with inner-selection thresholds of 140.875–143.500 μg/m³). Second, the *temporal evaluation threshold* was the 90th percentile of the 2022 test set (149.500 μg/m³), used only to define the high-ozone subgroup for post-hoc reporting. Third, the *spatial evaluation threshold* was a common full-sample 90th percentile (141.875 μg/m³), used solely to assign the pooled held-out predictions of all nine folds to one evaluation subgroup; it entered neither fitting, tuning nor weighting.

Because the temporal and spatial evaluation thresholds differ (149.500 vs 141.875 μg/m³), the two protocols evaluate their high-ozone subgroups at different concentration lines, and the corresponding percentage changes should not be read as a comparison at an identical threshold. As a sensitivity check, held-out predictions were regrouped with each fold's own training-period threshold (140.436–143.125 μg/m³), which produced 2,224 high-ozone rows (10.14%); the improvement was re-evaluated on that grouping without retraining.

### 2.8 Evaluation metrics and error decomposition

Accuracy is reported as RMSE, MAE, bias (prediction minus observation) and R², all in μg/m³ except R². Differences are defined as new minus old, so a negative ΔRMSE indicates improvement. Stratified metrics use the 90th and 95th percentiles of the observed MDA8 within each evaluation set. Because squared-error totals combine group size and group RMSE (SSE<sub>g</sub> = n<sub>g</sub> × RMSE<sub>g</sub>²), gains and costs are additionally decomposed into high-ozone (≥P90) and lower-ozone (<P90) squared-error changes; this separates "how much of the total error change came from the high-ozone group" from "what share of the total error the high-ozone group represents".

### 2.9 Sample-selection analysis

To test whether common-valid samples are representative, MDA8 was regressed on a common-valid indicator with station fixed effects and month-of-year fixed effects (1–12), using station-clustered standard errors. Year fixed effects were not included, so interannual differences in satellite observability are not removed and the coefficient remains a conditional association. The sample comprises the 50,192 station-days for which a satellite record existed and could be classified as common-valid or as having no common-valid pixel; the remaining 1,294 station-days had no satellite record and were excluded. Seasonal stratification and within-city comparisons were computed as descriptive checks. The coefficient is a conditional association, not a causal effect of satellite observation on ozone.

---

## 3. Results

### 3.1 Baseline estimation

Under the temporal protocol, the F1/XGBoost baseline reached RMSE 19.300 μg/m³, MAE 14.807 μg/m³ and R² 0.610 on the 4,691 common-valid 2022 rows. Under the nested LOCO protocol, the pooled RMSE was 17.277 μg/m³, MAE 13.330 μg/m³ and R² 0.645 (n = 21,925). The full-sample (51,486-row) reference run, which includes rows without common-valid pixels, gave RMSE 18.152 μg/m³ and R² 0.715 for 2022; it is reported only as a pipeline consistency check and is not compared with the common-valid results.

Adding the ground pollutant block to the meteorological block (F1 versus F0) reduced RMSE by 14.1% in the temporal test and by 15.0% in the LOCO test for XGBoost (Table S1), confirming that the conventional pollutant predictors, not the satellite columns, are the dominant non-meteorological source of information in this setting. This establishes the baseline against which the satellite ablation is measured.

### 3.2 Incremental value of satellite precursor columns

Table 1 summarizes the same-sample ablation. Adding HCHO alone reduced temporal RMSE by 0.398 μg/m³ (−2.06%) and spatial RMSE by 0.317 μg/m³ (−1.84%); adding NO₂ alone gave −0.355 μg/m³ (−1.84%) and −0.235 μg/m³ (−1.36%); adding both gave the largest reductions, −0.658 μg/m³ (−3.41%) in time and −0.477 μg/m³ (−2.76%) in space. Temporal MAE decreased from 14.807 to 14.365 μg/m³ and the 95th-percentile error from 40.363 to 38.052 μg/m³ for E2-3; spatial MAE decreased from 13.330 to 12.980 μg/m³ and R² rose from 0.645 to 0.664.

**Table 1.** Satellite ablation on the identical 21,925 common-valid station-days (E2-0 to E2-3). Δ = new − old; Δ% relative to E2-0. CIs are 95% percentile station-cluster bootstrap intervals for each version relative to E2-0.

| Version | Predictors | Temporal RMSE | ΔRMSE | Δ% | Temporal R² | Spatial RMSE | ΔRMSE | Δ% | Spatial R² | Temporal CI | Spatial CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E2-0 | F1 | 19.300 | — | — | 0.610 | 17.277 | — | — | 0.645 | — | — |
| E2-1 | F1 + HCHO | 18.902 | −0.398 | −2.06 | 0.626 | 16.960 | −0.317 | −1.84 | 0.658 | [−0.572, −0.213] | [−0.459, −0.184] |
| E2-2 | F1 + NO₂ | 18.945 | −0.355 | −1.84 | 0.624 | 17.042 | −0.235 | −1.36 | 0.654 | [−0.527, −0.194] | [−0.572, +0.077] |
| E2-3 | F1 + HCHO + NO₂ | 18.642 | −0.658 | −3.41 | 0.636 | 16.800 | −0.477 | −2.76 | 0.664 | [−0.930, −0.396] | [−0.852, −0.156] |

The direction of the E2-3 gain is consistent across protocols, but its strength is not. In the temporal test the station-cluster interval [−0.930, −0.396] and the city-cluster interval [−1.024, −0.287] both exclude zero. In the LOCO test the station-cluster interval [−0.852, −0.156] excludes zero, whereas the city-cluster interval [−1.212, +0.107] crosses zero. The spatial absolute reduction (−0.477 μg/m³) also falls short of the pre-specified −0.5 μg/m³ criterion, so the satellite block does not meet the retention rule. We therefore describe the outcome as conditional incremental value: a clear improvement in the independent-year test, and a weaker, city-dependent improvement in the cross-city test.

The two satellites are partly complementary rather than redundant. E2-1 and E2-2 each deliver a reduction of similar size, and E2-3 is better than either single-column version in both protocols, which is consistent with the low mutual correlation of the columns (Pearson r = 0.062, Spearman ρ = 0.132, n = 21,925). The supplementary ratio experiment did not change this picture: on its 21,223-row subsample, E2-4 gave a temporal reduction of 0.678 μg/m³ (−3.48%), close to E2-3 on the same subsample, but a spatial reduction of only 0.244 μg/m³ (−1.41%), again failing the retention rule. The ratio was thus not retained.

City-level results are heterogeneous. Relative to E2-0, the 5 km spatial gain ranged from −2.782 μg/m³ (Longyan) and −1.255 μg/m³ (Zhangzhou) to +0.913 μg/m³ (Xiamen) and +0.135 μg/m³ (Putian), with Ningde at −0.706 μg/m³. Two cities therefore became less accurate when the satellite block was added. Buffer width changed the magnitude but not the direction: the E2-3 versus E2-0 ΔRMSE at 3, 5 and 10 km was −0.583, −0.658 and −0.888 μg/m³ in time and −0.465, −0.477 and −0.712 μg/m³ in space (Table S1). The 5 km buffer, fixed before the experiments, remains the primary scale, and the 10 km result is not used to backfill the main conclusion.

### 3.3 How the model uses the satellite features

On the frozen E2-3 model, the two satellite features together account for 6.32% of the summed mean absolute SHAP contribution (Fig. 1). NO₂ (`no2_col`) has mean |SHAP| 2.284 μg/m³ (4.10%, rank 9 of 30) and HCHO (`hcho_col`) 1.232 μg/m³ (2.21%, rank 16). The leading predictors are the seasonal cycle (`doy_cos`, 6.14, 11.03%), PM₂.₅ (5.91, 10.61%) and PM₁₀ (5.24, 9.42%). Satellite information thus supplements, but does not displace, the ground and meteorological information.

The response patterns observed in the frozen model are as follows (Fig. 2). The decile-averaged SHAP contribution of HCHO rises monotonically from −1.702 μg/m³ in the lowest decile to +3.397 μg/m³ in the highest. NO₂ peaks in the eighth decile (+3.087 μg/m³) and then declines to +1.461 μg/m³ in the highest decile, indicating saturation rather than an unbounded response in the upper tail. The HCHO × NO₂ interaction has mean |SHAP interaction| 0.0509 μg/m³, an order of magnitude below the main effects (Fig. S1).

The satellite contribution is larger in high-ozone and summer samples. Mean |SHAP| for the two features is 3.298 μg/m³ below the median MDA8 and 4.691 μg/m³ above the 90th percentile, and the signed contribution shifts from −0.542 to +3.653 μg/m³ between these groups; the July–August absolute contribution is the highest of the four seasons (4.588 μg/m³) (Fig. S3). Across cities, the satellite share ranges from 5.51% to 9.73%, and the Spearman correlation between this share and the city-level ΔRMSE is +0.617 over nine cities (Fig. S2); with nine units this is reported as an exploratory description, not as a general relationship.

Two qualifications follow directly from these numbers and are maintained in the discussion. First, a larger SHAP contribution describes how the model uses a feature; it is not a measurement of the corresponding atmospheric process. Second, a larger contribution does not imply a smaller error: Xiamen has a positive signed satellite contribution (+2.615 μg/m³) but a higher RMSE (+0.913 μg/m³) after the satellite block is added, whereas Longyan has a negative signed contribution (−1.346 μg/m³) and the largest improvement (−2.782 μg/m³). We therefore do not link satellite reliance to error reduction.

### 3.4 High-ozone reweighting

Table 2 reports the reweighting experiment. In the temporal test, λ = 2 (E4-1) reduced overall RMSE from 18.642 to 18.270 μg/m³ (−1.99%) and high-ozone RMSE from 33.341 to 31.847 μg/m³ (−4.48%); high-ozone MAE fell from 28.960 to 27.172 μg/m³ (−6.17%) and high-ozone bias from −28.849 to −27.026 μg/m³, indicating a partial correction of the systematic underestimation at high concentrations. The lower-ozone subgroup also improved slightly (16.193 → 16.057 μg/m³, −0.84%). Both the station-cluster and city-cluster intervals for the high-ozone change exclude zero ([−1.977, −1.000] and [−1.993, −1.072] μg/m³).

**Table 2.** High-ozone reweighting (λ = 2) against the unweighted model in both protocols. Overall and subgroup RMSE in μg/m³; Δ% = (E4-1 − E4-0)/E4-0.

| Protocol | Version | Overall RMSE | Δ% | High-ozone RMSE (≥P90) | Δ% | Lower-ozone RMSE (<P90) | Δ% | High-ozone bias |
|---|---|---|---|---|---|---|---|---|
| Temporal (n = 4,691) | E4-0 | 18.642 | — | 33.341 (n = 471) | — | 16.193 | — | −28.849 |
| Temporal (n = 4,691) | E4-1 | 18.270 | −1.99 | 31.847 (n = 471) | −4.48 | 16.057 | −0.84 | −27.026 |
| Spatial (n = 21,925) | E4-0 | 16.800 | — | 23.282 (n = 2,198) | — | 15.916 | — | −17.731 |
| Spatial (n = 21,925) | E4-1 | 16.836 | +0.21 | 21.562 (n = 2,198) | −7.39 | 16.224 | +1.94 | −15.444 |

The spatial protocol shows a different balance. High-ozone RMSE fell from 23.282 to 21.562 μg/m³ (−7.39%; station-cluster [−2.068, −1.364], city-cluster [−2.357, −1.083]) and high-ozone bias improved from −17.731 to −15.444 μg/m³, with all nine cities showing a reduction in high-ozone RMSE (Fig. 5). Overall RMSE, however, rose from 16.800 to 16.836 μg/m³ (+0.21%), and the lower-ozone subgroup rose from 15.916 to 16.224 μg/m³ (+1.94%). The overall change is not distinguishable from zero (station-cluster [−0.134, +0.205], *P*<sub>improve</sub> = 0.336), so the correct statement is that no clear overall improvement or deterioration was observed, not that overall performance was preserved.

Squared-error decomposition explains how both outcomes arise. In the temporal test the high-ozone group contains 10.04% of the rows (471) but 32.1% of the E4-0 squared error; reweighting reduced that group's squared error by 4.59 × 10⁴ μg²/m⁶, which is 71.3% of the total reduction (6.44 × 10⁴ μg²/m⁶), while the lower-ozone group contributed the remaining reduction of 1.85 × 10⁴ μg²/m⁶. Here the high-ozone gain and the overall gain move in the same direction. In the spatial test the high-ozone group contains 10.03% of the rows (2,198) and 19.3% of the E4-0 squared error; reweighting reduced its squared error by 1.70 × 10⁵ μg²/m⁶ but raised that of the lower-ozone group by 1.96 × 10⁵ μg²/m⁶, so the net change is +2.62 × 10⁴ μg²/m⁶, about +0.4% of the baseline squared error, corresponding to the +0.21% RMSE change (Table S4). The 71.3% figure is a share of the total squared-error reduction, not a share of the total error.

The threshold sensitivity check confirms that the spatial high-ozone result does not hinge on the common post-hoc threshold. Regrouping the held-out predictions with each fold's own training-period threshold (2,224 rows, 10.14%) gave 23.273 → 21.536 μg/m³ (−7.47%), with nine of nine cities improving (Table S7). Weight strength behaves as expected only within the tested grid: λ = 1.5 reduced temporal high-ozone RMSE by 2.48% and spatial high-ozone RMSE by 3.94%, with spatial overall +0.57%; this is below the pre-registered temporal criterion and is therefore not presented as a parallel main configuration. We do not interpret the λ sequence as evidence of a general dose–response law.

### 3.5 Robustness and sample representativeness

Three robustness checks support the scope of the conclusions. First, buffer width (3, 5 and 10 km) preserved the sign of the satellite gain in both protocols, with the absolute magnitude increasing at 10 km (Table S1). Second, the high-ozone conclusion survived the fold-wise regrouping described above. Third, the algorithm controls behaved consistently: LightGBM reproduced the satellite direction in both protocols (spatial −0.403 μg/m³ for E2-3) and the high-ozone reweighting improved its high-ozone RMSE in both protocols, although with a larger overall cost than XGBoost (Table S1).

The representativeness check indicates a systematic difference between sample types. Common-valid station-days have a higher MDA8 than station-days with a satellite record but no common-valid pixel. The raw difference is +26.61 μg/m³; after controlling for station and month-of-year fixed effects, the conditional difference is +25.71 μg/m³ (station-clustered SE 0.65, t = 39.3, p < 0.001, n = 50,192, 38 clusters; Table S6). The direction is identical in all four seasons (+31.136, +31.241, +18.575 and +29.272 μg/m³ for winter, spring, summer and autumn) and in all nine cities (+19.23 to +31.81 μg/m³). Common-valid station-days are therefore not a random subset of the ground network, and both the satellite and reweighting results refer to satellite-observable conditions. The coefficient is a conditional association; it is not evidence that satellite observation causes higher ozone, and it cannot be attributed solely to cloud-free conditions.

Coverage is uneven in other respects as well. The high-ozone evaluation group contains 4.82%–17.75% of each city's rows, with 108–454 high-ozone rows per city across 3–6 stations (Table S3). "Nine cities in the same direction" therefore describes directional consistency, not independent significance in every city.

---

## 4. Discussion

### 4.1 Why the satellite gain is real but conditional

The E2 results show detectable incremental information from the two TROPOMI columns, but three limits are as important as the gain itself: the spatial absolute reduction (−0.477 μg/m³) falls below the pre-specified −0.5 μg/m³ criterion, the city-cluster bootstrap interval crosses zero, and the satellite features account for only 6.32% of the total SHAP contribution. The defensible conclusion is that the columns supplement information not covered by the ground and meteorological predictors, without changing the dominant source of information in the model.

Four possible explanations are consistent with this pattern. They are candidates, not separated by an independent experiment, and none is a statement about ozone chemistry; they are therefore phrased as possibilities.

1. **Vertical representativeness.** TROPOMI measures the column, whereas MDA8 reflects near-surface photochemistry and boundary-layer mixing. Neither the vertical distribution within the boundary layer nor the free-tropospheric contribution can be recovered from a single column value. This may simultaneously explain why the columns carry independent information and why their share remains modest.
2. **Sampling imposed by common-valid screening.** Only 42.58% of station-days are common-valid, and that subset has a higher MDA8 (Section 3.5). The model therefore learns the mapping under satellite-observable conditions, which depend on cloud screening, retrieval quality, spatial coverage and pixel counts and may favour comparatively cloud-free, high-radiation conditions. This should limit extrapolation to other weather types (Table S6).
3. **Space–time matching.** A daily concatenated product typically provides one or two overpasses per day, whereas MDA8 integrates eight hours; a 5 km buffer aggregation further smooths the column field. Scale mismatch can dilute a signal that would otherwise be usable.
4. **Limited but non-zero information overlap.** In the common-valid sample, column NO₂ correlates with ground NO₂ at r = 0.551 (Spearman 0.526) and with PM₁₀ at 0.407, while HCHO correlates weakly with all ground pollutants (|r| ≤ 0.09). Standard VIFs, computed with all other 28 F1 features plus the other satellite feature in the auxiliary regression, are 2.095 for NO₂ and 1.295 for HCHO; adding the other satellite feature did not change these values at three decimal places, so no appreciable additional collinearity is introduced beyond the F1 covariates (Table S5b). The columns themselves are weakly correlated (Pearson r = 0.062, Spearman ρ = 0.132, n = 21,925). Neither diagnostic indicates strong collinearity (Table S5). Information overlap can therefore only partly limit the NO₂ contribution and cannot explain the low HCHO share; the latter is more plausibly related to the signal-to-noise ratio of the HCHO column in this subset and to the matching scale, but that remains an untested explanation. We do not use "high collinearity" as a general account of the modest gain.

The NO₂ share (4.10%) exceeds the HCHO share (2.21%), but this difference must not be read as evidence of a more direct or stronger chemical link between NO₂ and ozone. The columns differ in lifetime, source structure and seasonality, and each overlaps differently with the corresponding ground pollutant; without VOC/PAN measurements and vertical profiles, the controlling regime of ozone production cannot be inferred from these shares.

### 4.2 Model attribution is not error reduction

Within the frozen model and sample, SHAP provides two structural descriptions. First, the decile-averaged HCHO contribution rises monotonically, whereas the NO₂ contribution peaks and then declines, indicating saturation at high column values. Second, the satellite absolute contribution is larger in high-ozone samples (4.691 versus 3.298 μg/m³) and in summer (4.588 μg/m³), and the signed contribution switches from negative to positive between the low and high groups. These describe how the model uses the features, and they are consistent with the possibility that precursor information is used more heavily in high-ozone conditions; they are not measurements of atmospheric processes and say nothing directly about predictive accuracy. A chemistry-informed interpretation framework (Sun et al., 2026) makes the same point: explanations must be anchored in chemical knowledge before they are read as mechanisms.

The distinction matters because SHAP magnitude and error change are not interchangeable. In the city-level results, Xiamen has a positive signed satellite contribution (+2.615 μg/m³) but a higher RMSE (+0.913 μg/m³), whereas Longyan has a negative contribution (−1.346 μg/m³) and the largest improvement (−2.782 μg/m³). The Spearman correlation of +0.617 between satellite share and city ΔRMSE is reported as an exploratory description over nine cities and is not used to establish a general rule.

Regional heterogeneity also requires a limited interpretation. The satellite share varies only from 5.51% to 9.73% across cities, whereas the spatial gain varies from −2.782 to +0.913 μg/m³. Plausible contributors include station density and spatial representativeness, differences in common-valid coverage, background pollution levels and coastal–inland contrasts in air-mass origin and boundary-layer structure. These factors cannot be separated with the present data, so the heterogeneity is reported as a fact rather than attributed to a single mechanism, and the earlier "inland better than coastal" framing is not adopted.

### 4.3 High-ozone reweighting: a targeted gain with an interval trade-off

Reweighting improved the high-ozone group in both protocols: −4.48% in the temporal test and −7.39% in the spatial test, with cluster intervals excluding zero and a reduction in the systematic high-ozone underestimation. Nine of nine cities showed a lower high-ozone RMSE. Why, then, does overall spatial performance not improve?

The answer lies in the mismatch between error structure and sample structure, which the squared-error decomposition makes explicit.

1. **High values carry larger unit errors, but they are not the majority of the total error.** In the temporal test the high-ozone group is 10.04% of the rows but 32.1% of the baseline squared error. Reweighting reduces that group's squared error by 4.59 × 10⁴ μg²/m⁶, which is 71% of the total reduction, while the remaining samples also improve. Here the targeted gain and the overall gain coincide.
2. **In the spatial test the gain is offset by a small cost elsewhere.** The high-ozone group is 10.03% of the rows and 19.3% of the baseline squared error; its squared error falls by 1.70 × 10⁵ μg²/m⁶, but the lower-ozone group's rises by 1.96 × 10⁵ μg²/m⁶, giving a net +2.62 × 10⁴ μg²/m⁶ and +0.21% in RMSE. The paired interval crosses zero, so the appropriate wording is that no clear overall change was observed.
3. **The bias direction shifts.** Spatial overall bias moves from +1.220 to +2.208 μg/m³ while high-ozone bias improves from −17.731 to −15.444 μg/m³: the model shifts slightly upward in exchange for a smaller underestimation at high concentrations. The strategy is thus a trade-off between concentration intervals rather than a one-directional improvement.

The fold-wise threshold regrouping (−7.47%, nine of nine cities) shows that this conclusion does not depend on the choice of a common post-hoc threshold. Within the tested weight grid (λ = 1, 1.5, 2), λ = 2 produced the largest high-ozone gain; λ = 1.5 improved the temporal high-ozone group by only 2.48%, below the pre-registered 3% criterion, and is not presented as a parallel main configuration. This observation is limited to the tested grid and does not establish a general monotonic dose–response relationship.

Finally, the nature of E4 should be stated plainly: it is a validated training strategy, in which a reweighted loss amplifies the relative influence of high-value samples. Its benefit comes from the objective function, not from any representation of ozone formation, and describing it as a mechanistic improvement would exceed what the experiment supports.

### 4.4 Relation to previous ozone-estimation studies

This section compares tasks and validation protocols first and performance values second. Studies that differ in period, region, target definition (MDA8 versus daily mean) or validation split are marked as not directly comparable, and the present values are not used to claim superiority.

**Gap 1 — input level: the marginal gain of satellite precursors is not homogeneous.** Previous studies typically add OMI/TROPOMI columns directly as ozone predictors (Chen J. et al., 2022; Chen X. et al., 2024) or use retrieved surface precursors and satellite ozone columns (Wang Y. et al., 2021; Wang W. et al., 2022), but the reported gains differ: Wang W. et al. (2022) found that satellite data mainly improved extreme-value estimation, whereas Chen X. et al. (2024) obtained mainly a coverage gain with accuracy close to a satellite-free configuration. A gain reported elsewhere therefore cannot be adopted as a prior for this region. Our E2 ablation holds rows, split and model fixed and yields −3.41% (temporal) and −2.76% (spatial), which are relative gains within one protocol and are consistent with the "limited and high-value-oriented" direction of the literature rather than a reproduction or an improvement of published values.

**Gap 2 — validation level: random and spatial hold-out answer different questions.** Sample-level or random cross-validation mainly tests interpolation within a population, whereas spatial hold-out tests extrapolation to unseen locations. Chen G. et al. (2021) reported R² 0.84 under sample-level CV and 0.79 under station-level CV for the same model; Young et al. (2016) showed that conventional K-fold and leave-one-out schemes overestimate performance away from monitoring sites; Roberts et al. (2017) provided the methodological basis for blocked cross-validation. We therefore do not compare absolute values with studies reporting R² near 0.9 under random CV, and we compare only relative gains within a protocol. The nested LOCO design, in which the held-out city influences neither tuning nor thresholds, supports within-province cross-city applicability; it does not establish transferability across provinces, climate zones or satellite-observing conditions.

**Gap 3 — objective level: overall optimum is not high-value optimum.** Most studies report a single overall RMSE/R² or an exceedance-classification score, and rarely report stratified metrics together with the overall cost. Deng T. et al. (2022) emphasized high-ozone errors through a weighted loss, but the task was 1–3-day forecasting and the cost below the high-value range was not reported alongside. We report gain and cost together in an estimation task: temporal overall and high-ozone performance improve simultaneously, whereas in the spatial protocol high-ozone RMSE falls by 7.39% while the lower-ozone group rises by 1.94% and overall changes by +0.21%. The squared-error decomposition shows that the spatial net change is the residual of two offsetting group changes. The contribution is therefore stratified reporting and explicit trade-off disclosure, not the invention of high-value weighting.

**Relation to recent regional studies.** The three gaps are visible in the latest literature. National-scale stacked and deep-learning products (Li Z. et al., 2024; Li T. et al., 2025) and satellite-fusion products (Wang Y. et al., 2025; Wang L. et al., 2025) target 24-h or daily national grids and report overall accuracy, so their values are not comparable with the MDA8, station-level, province-scale values here; what they do establish is that fusion itself is no longer a novelty. Episode-oriented analyses using the same TROPOMI precursors (Zhu et al., 2024) show that the columns carry information about ozone enhancement, whereas city-scale multifactor analyses (Wang and Yu, 2026) and trajectory-based sensitivity studies (Hu et al., 2026) pursue attribution questions that this study does not attempt. Our contribution is therefore not an accuracy claim relative to these studies but the three conditional questions above.

**Positioning.** These three gaps correspond to the three contributions of this study: a same-protocol evaluation of satellite incremental value, with the nested LOCO protocol as its spatial test; a quantification of satellite-sample representativeness; and a gain–cost decomposition of a high-ozone strategy. None supports a "best-performing model" claim; the SHAP analysis is used only to describe model behaviour (Lundberg & Lee, 2017; Kumar et al., 2020) and is not treated as chemical evidence. The full comparison matrix, including task, satellite input, validation protocol and reported metrics, is given in Table S8.

### 4.5 Future work

Systematic limitations are listed in Section 5. Future work is proposed only where a clear scientific question exists:

1. **All-weather coverage.** Use AOD, reanalysis or model output to fill dates without satellite pixels, and test whether the conclusions extend beyond satellite-observable conditions. No imputation or extrapolation is performed at present.
2. **Chemical observations.** VOC, PAN or vertical profile measurements would be required before discussing the controlling regime of ozone production; until then no such inference is made.
3. **Transport processes.** HYSPLIT or equivalent trajectory analysis should be introduced only if a specific transport question can be posed (for example, whether the coastal–inland gain difference is explained by air-mass origin) and a defensible clustering scheme is available.
4. **Temporal extrapolation.** This study is a same-period estimation; forecasting would require a separate design with strictly prospective splits.

## 5. Limitations

1. **Satellite-observable sampling.** Common-valid samples cover 42.58% of the modeling table (21,925/51,486 station-days). All conclusions hold only under satellite-observable conditions; cloudy and clear days differ in ozone distribution, so the results are not an unbiased estimate for all periods.
2. **Sample-selection bias.** Common-valid station-days have systematically higher MDA8: seasonal stratification, within-city comparison and the station plus month-of-year fixed-effect regression (β ≈ +25.71 μg/m³, SE 0.65, station-clustered, n = 50,192, 38 clusters) all point in the same direction (Table S6). The regression controls station and month-of-year effects but not year effects, so interannual differences in satellite observability are not fully removed. Results should not be extrapolated to all weather types.
3. **Limited chemical observations.** Without VOC or PAN measurements, VOC-limited and NOx-limited regimes cannot be distinguished; HCHO and NO₂ contributions in the model are not chemical evidence.
4. **Uneven spatial representation.** Station counts, common-valid coverage and high-ozone sample sizes differ across cities (108–454 high-ozone rows from 3–6 stations; Table S3). "Nine cities in the same direction" does not mean independent significance in each city. In addition, the spatial high-ozone subgroup is defined post hoc by the full-sample P90 (141.875 μg/m³). This threshold entered neither fitting, tuning nor weighting, but it depends on the distribution of the evaluation period; the fold-wise regrouping check (−7.47%) shows that the direction of the conclusion is unchanged (Table S7).
5. **Interpretation limits.** SHAP reflects predictive attribution in the model. Attribution magnitude and error improvement are not necessarily related, as the Xiamen and Longyan results show.
6. **Task boundary.** TROPOMI and MDA8 are same-day information; this study is a near-surface ozone estimation (mapping) study, not a forecasting study. The terms estimation and mapping are used consistently throughout.

## 6. Conclusions

1. A stable MDA8 estimation baseline can be established for Fujian Province from CNEMC observations, ERA5 meteorology and multi-source pollutant data: temporal RMSE 19.300 μg/m³ (R² = 0.610) and nested LOCO RMSE 17.277 μg/m³ (R² = 0.645) on the common-valid sample, providing a single reproducible reference for the incremental and targeted experiments.
2. TROPOMI HCHO and NO₂ provide limited but detectable incremental information: temporal RMSE falls by 3.41% and spatial RMSE by 2.76% (−0.477 μg/m³, below the pre-specified −0.5 μg/m³ criterion, with the city-cluster interval crossing zero). The gain is conditional on city and on satellite-observable sampling, not a general improvement.
3. SHAP attributes 6.32% of the total contribution to the two satellite features, with larger contributions in high-ozone and summer samples; the HCHO contribution rises monotonically across deciles and the NO₂ contribution saturates at high values, with a very small interaction. These describe model behaviour and do not constitute evidence about ozone formation, nor do they map directly onto error changes.
4. High-ozone reweighting (λ = 2) reduces high-ozone RMSE by 4.48% in the temporal test and 7.39% in the nested spatial test, with station-level intervals excluding zero; spatial overall RMSE changes by +0.21% (interval crossing zero) and the lower-ozone group rises by 1.94%. The method therefore delivers a targeted improvement for the high-concentration range at the cost of an interval trade-off, and cannot be described as a general improvement across all concentration ranges.

Together, the three contributions — conditional incremental value of the satellite precursors, quantified representativeness of satellite-observable samples, and the high-ozone gain–cost trade-off — form one application-methods framework in which each element is conditional in scope: the first is constrained by common-valid screening and city heterogeneity, the second defines the conditions under which the estimates apply, and the third by the trade-off between concentration intervals.

---

## Data availability

The ground ozone observations are available from the Zenodo record 10.5281/zenodo.10911197 (CNEMC hourly surface ozone across China). Daily ground pollutant concentrations are available from the public CNEMC historical air-quality archive (quotsoft). ERA5 and ERA5-Land meteorology were obtained through the Open-Meteo Historical Weather API. TROPOMI HCHO and NO₂ column products are available from the Copernicus Data Space Ecosystem. The processed modeling table, frozen result tables and analysis code are archived with the project and are available from the corresponding author upon reasonable request; a public repository DOI will be provided before publication.

## Code availability

All experiments were run with fixed random seeds and frozen result archives verified by SHA-256 checksums. The analysis scripts (feature construction, temporal and nested spatial validation, ablation, SHAP and reweighting) are available from the corresponding author upon reasonable request.

## CRediT authorship contribution statement

**[Author 1]**: Conceptualization, Data curation, Formal analysis, Methodology, Software, Validation, Visualization, Writing – original draft. **[Author 2]**: Data curation, Investigation, Methodology, Writing – review & editing. **[Author 3]**: Supervision, Resources, Writing – review & editing. **[Corresponding author]**: Conceptualization, Funding acquisition, Project administration, Supervision, Writing – review & editing.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgements

We thank the China National Environmental Monitoring Centre for the ground ozone observations, the ECMWF and Open-Meteo teams for reanalysis access, and the European Space Agency and the Copernicus programme for the Sentinel-5P TROPOMI products. This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## References

Chen G., Chen J., Dong G., Yang J., Liu Y., Zhang Y., 2021. Improving satellite-based estimation of surface ozone across China during 2008–2019 using iterative random forest model and high-resolution grid meteorological data. Sustainable Cities and Society 69, 102807. https://doi.org/10.1016/j.scs.2021.102807

Chen J., Shen H., Li X., Li T., Wei Y., 2022. Ground-level ozone estimation based on geo-intelligent machine learning by fusing in-situ observations, remote sensing data, and model simulation data. International Journal of Applied Earth Observation and Geoinformation 112, 102955. https://doi.org/10.1016/j.jag.2022.102955

Chen T., Guestrin C., 2016. XGBoost: A scalable tree boosting system. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785–794. https://doi.org/10.1145/2939672.2939785

Chen X., et al., 2024. Estimation of near-surface O₃ concentration based on reconstructed TROPOMI NO₂, CO and HCHO data (in Chinese). National Remote Sensing Bulletin 28 (9), 2348–2361. https://doi.org/10.11834/jrs.20233068

Deng T., Lin H., Chen X., et al., 2022. Clustering-based spatial transfer learning for short-term ozone forecasting. Journal of Hazardous Materials Advances 8, 100168. https://doi.org/10.1016/j.hazadv.2022.100168

Hersbach H., Bell B., Berrisford P., Hirahara S., Horányi A., Muñoz-Sabater J., et al., 2020. The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society 146 (730), 1999–2049. https://doi.org/10.1002/qj.3803

Hickman S., Kelp M., Griffiths P., Doerksen K., Miyazaki K., Pennington E., et al., 2025. Applications of machine learning and artificial intelligence in tropospheric ozone research. Geoscientific Model Development 18, 8777–8800. https://doi.org/10.5194/gmd-18-8777-2025

Hu B., Gao Y., Chen N., Zeng J., Chen S., Zeng Z., et al., 2026. Quantifying transport contributions and diagnosing ozone formation sensitivity: An integrated approach using machine learning with high-resolution gridded data and backward trajectories. Journal of Cleaner Production 540, 147524. https://doi.org/10.1016/j.jclepro.2026.147524

Ke G., Meng Q., Finley T., Wang T., Chen W., Ma W., Ye Q., Liu T.-Y., 2017. LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems 30, 3146–3154.

Kumar I.E., Venkatasubramanian S., Scheidegger C., Friedler S., 2020. Problems with Shapley-value-based explanations as feature importance measures. Proceedings of the 37th International Conference on Machine Learning, PMLR 119, 5491–5500. https://arxiv.org/abs/2002.11097

Li S., Song G., Xing J., Dong J., Zhang M., Fan C., et al., 2025. Unraveling overestimated exposure risks through hourly ozone retrievals from next-generation geostationary satellites. Nature Communications 16. https://doi.org/10.1038/s41467-025-58652-2

Li T., Wu J., Wang Y., Su Y., 2025. Improved seamless mapping of surface O₃ concentrations using an integrated deep learning framework. npj Climate and Atmospheric Science 8. https://doi.org/10.1038/s41612-025-01007-x

Li Z., Wang W., He Q., Chen X., Huang J., Zhang M., 2024. Estimating ground-level high-resolution ozone concentration across China using a stacked machine-learning method. Atmospheric Pollution Research 15, 102114. https://doi.org/10.1016/j.apr.2024.102114

Lundberg S.M., Lee S.-I., 2017. A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems 30, 4765–4774. https://arxiv.org/abs/1705.07874

Lundberg S.M., Erion G., Chen H., DeGrave A., Prutkin J.M., Nair B., Katz R., Himmelfarb J., Bansal N., Lee S.-I., 2020. From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence 2 (1), 56–67. https://doi.org/10.1038/s42256-019-0138-9

Roberts D.R., Bahn V., Ciuti S., Boyce M.S., Elith J., Guillera-Arroita G., et al., 2017. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography 40 (8), 913–929. https://doi.org/10.1111/ecog.02881

Sun Y., Ding S., Wang C., Liu D., 2026. Machine learning for ozone pollution research: A chemistry-informed framework for model selection and interpretation. ACS Earth and Space Chemistry 10, 2201–2220. https://doi.org/10.1021/acsearthspacechem.6c00213

Tang D., Zhan Y., Yang F., 2024. A review of machine learning for modeling air quality: Overlooked but important issues. Atmospheric Research 300, 107261. https://doi.org/10.1016/j.atmosres.2024.107261

Veefkind J.P., Aben I., McMullan K., Förster H., de Vries J., Otter G., et al., 2012. TROPOMI on the ESA Sentinel-5 Precursor: A GMES mission for global observations of the atmospheric composition for climate, air quality and ozone layer applications. Remote Sensing of Environment 127, 70–82. https://doi.org/10.1016/j.rse.2011.09.027

Wang K., Yu X., 2026. Unraveling the multifactor predictors of urban ozone in China: A machine learning framework for synergistic pollution and greenhouse gas mitigation. Environmental Science & Technology 60, 26370–26381. https://doi.org/10.1021/acs.est.6c08096

Wang L., Yang X., Zhou H., Yang Y., Dong J., Dong S., et al., 2025. Reanalysis-based retrieval of near-surface ozone concentrations and its implications for ozone pollution events across China. Atmospheric Pollution Research 16, 102643. https://doi.org/10.1016/j.apr.2025.102643

Wang W., Liu X., Bi J., Liu Y., 2022. A machine learning model to estimate ground-level ozone concentrations in California using TROPOMI data and high-resolution meteorology. Environment International 158, 106917. https://doi.org/10.1016/j.envint.2021.106917

Wang Y., Yuan Q., Li T., Zhu L., Zhang L., et al., 2021. Estimating daily full-coverage near surface O₃, CO, and NO₂ concentrations at a high spatial resolution over China based on S5P-TROPOMI and GEOS-FP. ISPRS Journal of Photogrammetry and Remote Sensing 175, 311–325. https://doi.org/10.1016/j.isprsjprs.2021.03.018

Wang Y., Chen B., Ye Q., Zhao L., Song Z., 2025. Estimation and impact factor analysis of 24-h near-surface ozone concentration in China using FY-4A/B collaboration and machine learning. Atmospheric Pollution Research 16, 102538. https://doi.org/10.1016/j.apr.2025.102538

Young M.T., Bechle M.J., Sampson P.D., Szpiro A.A., Marshall J.D., Sheppard L., Kaufman J.D., 2016. Satellite-based NO₂ and model validation in a national prediction model based on universal kriging and land-use regression. Environmental Science & Technology 50 (7), 3686–3694. https://doi.org/10.1021/acs.est.5b05099

Zhu B., Fang J., Zhang Y., Qiu J., Chen K., Zhang K., et al., 2024. Machine learning elucidates ubiquity of enhanced ozone air pollution in China linked to the spring festival effect. Atmospheric Pollution Research 15, 102127. https://doi.org/10.1016/j.apr.2024.102127

---

## Main tables and supplementary lists

**Table 3.** Thresholds used in the reweighting experiment and their roles.

| Purpose | Computed from | Value (μg/m³) | Used for fitting/weighting |
|---|---|---|---|
| Weighting threshold, temporal selection | Common-valid rows, 2019–2020 | 142.286 | Yes (selection stage only) |
| Weighting threshold, temporal refit | Common-valid rows, 2019–2021 | 139.750 | Yes (final temporal model) |
| Weighting threshold, outer spatial folds | Eight training cities, 2019–2022 | 140.436–143.125 | Yes (per fold) |
| Weighting threshold, inner selection | Eight training cities, 2019–2020 | 140.875–143.500 | Yes (selection stage only) |
| Temporal evaluation threshold | 2022 test set | 149.500 | No (post-hoc subgroup only) |
| Spatial evaluation threshold | All common-valid rows | 141.875 | No (post-hoc subgroup only) |

## Supplementary tables

| Table | Content | File |
|---|---|---|
| Table S1 | Buffer sensitivity (3/5/10 km), city-level satellite gains, algorithm controls | `table_supplementary_sensitivity.md` |
| Table S2 | SHAP contribution summary | `table_supplementary_shap.md` |
| Table S3 | Common-valid coverage and group composition | `table_coverage.md` |
| Table S4 | Squared-error decomposition of the reweighting experiment | `table_error_decomposition.md` |
| Table S5 | Satellite–ground correlations | `table_satellite_collinearity.md` |
| Table S5b | Standard VIF for the two satellite features (with F1-only diagnostic) | `table_satellite_vif.md` |
| Table S6 | Sample-selection regression and seasonal/city stratification | `table_supplementary_selection_bias.md` |
| Table S7 | Fold-wise threshold sensitivity of the spatial high-ozone result | `table_spatial_foldwise_threshold.md` |
| Table S8 | Literature comparison matrix (task, input, protocol, reported metrics) | `table_literature_matrix.md` |

## Figure captions and numbering

| Figure | Caption | File (renamed copy) | Original file |
|---|---|---|---|
| Fig. 1 | Global mean absolute SHAP contributions in the frozen E2-3 model (2022 test, n = 4,691). | `outputs/paper/figures/Fig1.png` | `outputs/figures/e2_shap/fig1_global.png` |
| Fig. 2 | Decile dependence of the HCHO and NO₂ SHAP contributions. | `outputs/paper/figures/Fig2.png` | `outputs/figures/e2_shap/fig2_dependence.png` |
| Fig. 3 | Overall performance before and after high-ozone reweighting. | `outputs/paper/figures/Fig3.png` | `outputs/e4/figures/Fig11_e4_overall_performance.png` |
| Fig. 4 | High-ozone performance before and after reweighting. | `outputs/paper/figures/Fig4.png` | `outputs/e4/figures/Fig12_e4_high_ozone_performance.png` |
| Fig. 5 | City-level high-ozone effect of reweighting. | `outputs/paper/figures/Fig5.png` | `outputs/e4/figures/Fig13_e4_city_heterogeneity.png` |
| Fig. S1 | HCHO × NO₂ SHAP interaction. | `outputs/paper/figures/FigS1.png` | `outputs/figures/e2_shap/fig3_interaction.png` |
| Fig. S2 | City-level satellite SHAP share versus city ΔRMSE. | `outputs/paper/figures/FigS2.png` | `outputs/figures/e2_shap/fig4_city.png` |
| Fig. S3 | Satellite SHAP contribution by ozone subgroup and season. | `outputs/paper/figures/FigS3.png` | `outputs/figures/e2_shap/fig5_ozone.png` |
| Fig. S4 | Error distribution before and after reweighting. | `outputs/paper/figures/FigS4.png` | `outputs/e4/figures/Fig14_e4_error_distribution.png` |
