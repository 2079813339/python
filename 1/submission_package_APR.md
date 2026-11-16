# APR submission package

> Companion file to `outputs/paper/manuscript_APR_en.md`. Everything below is submission-ready
> except the bracketed placeholders, which require author input.

## 1. Article metadata

| Item | Value |
|---|---|
| Target journal | Atmospheric Pollution Research (Elsevier, ISSN 1309-1042) |
| Article type | Original research article |
| Title | Satellite precursor columns in regional near-surface ozone estimation: conditional incremental value, sample selection, and high-ozone reweighting in Fujian Province, China |
| Keywords (≤6) | near-surface ozone; TROPOMI; XGBoost; high-ozone reweighting; nested leave-one-city-out validation; Fujian Province |
| Abstract length | 238 words (as verified in the frozen abstract) |
| Main figures | 5 (Fig. 1—Fig. 5) |
| Main tables | 3 (Table 1—Table 3) |
| Supplementary figures | 4 (Fig. S1—Fig. S4) |
| Supplementary tables | 9 (Table S1—Table S8, including S5b) |
| References | 15 |

## 2. Highlights (optional for this journal; each ≤85 characters)

1. Same-sample ablation isolates the incremental value of HCHO and NO₂ columns.
2. Satellite gains are clear in time but weak and city-dependent in space.
3. Common-valid samples are biased toward higher ozone (+25.71 μg/m³).
4. High-ozone reweighting cuts high-ozone RMSE by 4.48–7.39%.
5. The spatial gain is offset by a +0.21% overall RMSE trade-off.

## 3. Cover letter

Dear Editor-in-Chief,

We submit the manuscript "Satellite precursor columns in regional near-surface ozone estimation:
conditional incremental value, sample selection, and high-ozone reweighting in Fujian Province,
China" for consideration as an original research article in *Atmospheric Pollution Research*.

Regional ozone mapping increasingly fuses ground monitoring, reanalysis meteorology and
satellite columns, yet two questions are rarely separated: whether satellite precursors add
information beyond the conventional predictors, and whether that information improves the
high-ozone range where the largest unit errors occur. Our study addresses both for Fujian
Province (2019–2022), using 21,925 satellite-observable station-days, a temporally independent
test year and a nested leave-one-city-out design.

Three features may be of interest to the journal's readership. First, the satellite ablation
uses identical rows, splits and model settings across four predictor sets, so gains can be
attributed to the columns rather than to sample composition; we report a clear temporal
improvement (−3.41%) and a weak, city-dependent spatial improvement (−2.76%, below our pre-specified
retention threshold). Second, we quantify the representativeness of the samples on which any
satellite-enhanced estimate is possible: after controlling for station and month, common-valid
station-days have a conditional MDA8 difference of +25.71 μg/m³, which bounds the scope of the
conclusions. Third, high-ozone reweighting reduces high-ozone RMSE by 4.48% (temporal) and 7.39%
(spatial) but raises lower-ozone RMSE by 1.94% in the spatial protocol, so we report the gain and
its cost together rather than claiming a general improvement.

The study is an application-methods contribution. It does not propose a new algorithm and does
not interpret model explanations as chemical evidence; without VOC/PAN measurements no
photochemical-regime inference is made. All results refer to estimation under satellite-observable
conditions, not to forecasting.

The manuscript is not under consideration elsewhere, all authors have approved the submission,
and we declare no competing interests. Data sources are public, and the processed modeling table
and analysis code are available from the corresponding author; a repository DOI will be provided
before publication.

Sincerely,
[Corresponding author], on behalf of all authors
[Affiliation, email, date]

## 4. Suggested reviewer areas (to be completed by the authors)

We suggest reviewers with expertise in (i) satellite remote sensing of trace gases and surface
ozone mapping, (ii) machine-learning validation design for geospatial data, and (iii)
regional air-quality monitoring in China. Names and contacts are intentionally left to the
authors, in line with journal policy on non-preferential suggestions.

## 5. Declarations to paste into the submission system

**Data availability.** Ground ozone observations: Zenodo record 10.5281/zenodo.10911197.
Ground pollutant daily means: public CNEMC historical air-quality archive (quotsoft). Meteorology:
ERA5/ERA5-Land via the Open-Meteo Historical Weather API. TROPOMI HCHO and NO₂ columns:
Copernicus Data Space Ecosystem. Processed data and code: available from the corresponding author
upon reasonable request; repository DOI to be inserted before publication.

**Code availability.** Analysis scripts (feature construction, temporal and nested spatial
validation, ablation, SHAP, reweighting) plus frozen result tables verified by SHA-256 checksums
are available from the corresponding author upon reasonable request.

**Competing interests.** None declared.

**Funding.** [To be completed.]

**CRediT.** As in the manuscript; author names to be inserted.

**Use of generative AI.** [Declare here if the journal requires it, e.g. "Generative AI tools
were used for language editing; all scientific content, analyses and interpretations are the
authors' own."]

## 6. File inventory

| File | Role | Status |
|---|---|---|
| `outputs/paper/manuscript_APR_en.md` | Full manuscript (Title → References → table/figure lists) | Draft complete; author placeholders |
| `outputs/paper/figures/Fig1.png` … `Fig5.png` | Main figures, unified numbering | Copied and renamed |
| `outputs/paper/figures/FigS1.png` … `FigS4.png` | Supplementary figures | Copied and renamed |
| `outputs/paper/table_supplementary_sensitivity.md` | Table S1 | Exists |
| `outputs/paper/table_supplementary_shap.md` | Table S2 | Exists |
| `outputs/paper/table_coverage.md` | Table S3 | Exists |
| `outputs/paper/table_error_decomposition.md` | Table S4 | Exists |
| `outputs/paper/table_satellite_collinearity.md` | Table S5 | Exists |
| `outputs/paper/table_satellite_vif.md` | Table S5b | Exists |
| `outputs/paper/table_supplementary_selection_bias.md` | Table S6 | Exists |
| `outputs/paper/table_spatial_foldwise_threshold.md` | Table S7 | Exists |
| `outputs/paper/table_literature_matrix.md` | Table S8 | Exists |

## 7. Items that still require author input

1. Author names, affiliations, corresponding author and email.
2. Funding statement (or explicit "no specific funding").
3. Repository DOI for code and processed data (or confirmation that "available on request" is
   acceptable to the editors).
4. Optional graphical abstract (not mandatory for this journal).
5. Final check of JCR and CAS (中科院) partition lists for the submission year; neither is
   provided on the journal website and both must be verified separately.
6. Formatting pass in the journal's Word template (reference style, line numbering, double
   spacing) at upload time.
