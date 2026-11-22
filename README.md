# SatEmis

卫星排放监测平台：TROPOMI 柱浓度审查 → 缺失重构 → 排放反演 → 清单比对 → 地面站交叉比对 → 可视化导出。

* 需求与实现计划：[docs/implementation_plan.md](docs/implementation_plan.md)
* 部署指南：[docs/deployment.md](docs/deployment.md) · 数据接入：[docs/data_ingestion.md](docs/data_ingestion.md)
* 技术栈：Python 3.11 + FastAPI + xarray/scipy + XGBoost/LightGBM + PyTorch(可选) + React 18 + deck.gl
* 当前状态：**后端 95 测试 + 前端 25 测试全绿 / 后端覆盖率 74.30% / ruff 全绿 / mypy 69 源文件 0 error**

---

## 一、快速开始（离线，无需凭证）

```powershell
# 1) 创建虚拟环境并安装（含开发依赖）
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# 2) 跑一遍测试验证安装
.\.venv\Scripts\python.exe -m pytest tests/unit tests/integration -m "not ml"

# 3) 启动后端 API
.\.venv\Scripts\python.exe -m uvicorn satemis.api.main:app --port 8000

# 4) 启动前端（另开一个终端）
npm --prefix frontend install
npm --prefix frontend run dev
# 浏览器访问 http://localhost:5173
```

默认 `SATEMIS_DATA_SOURCE=synthetic`，全流程用内置合成数据源（含可复现随机种子），**不需要任何 Sentinel-5P / ERA5 / CDS 凭证**，5 分钟内即可看到完整管道跑通。

切换到真实数据时设 `SATEMIS_DATA_SOURCE=real` 并配置 `.env`（参考 `.env.example`）。

---

## 二、CLI 命令参考

安装后通过 `satemis` 或 `python -m satemis.cli` 调用。

### 2.1 `satemis info` — 查看当前配置

```powershell
satemis info
```
打印解析后的 Settings（数据源、网格、ML 超参、存储后端等）+ 各通道能力矩阵。可用于诊断 env 是否生效。

### 2.2 `satemis channels` — 列出已注册通道

```powershell
satemis channels
```
列出 NO₂ 主通道 / CO 通道 / O₃ 通道及其元信息。

### 2.3 `satemis run` — 跑完整管道

```powershell
satemis run --species NO2 --start 2026-03-01 --end 2026-03-08 --channel main
```

完整参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--species` | `NO2` | 目标物种（NO2 / CO / O3 / HCHO / SO2 / CH4） |
| `--start` | 7 天前 | 起始日期 `YYYY-MM-DD` |
| `--end` | 昨天 | 结束日期 `YYYY-MM-DD` |
| `--channel` | `main` | 重构通道名 |
| `--no-validation` | off | 跳过 RF 独立验证（F-06） |
| `--no-inversion` | off | 跳过排放反演 + EMG + 清单比对 |
| `--no-train` | off | 跳过 ML 训练（用预训练模型） |
| `--ground-truth` | off | **开启地面站交叉比对（TCCON/Pandora）** |

输出：完整 `PipelineResult.summary()` JSON，含 `metrics / audit / quality / validation / inversion / emg / bayesian / ground_truth / reference / artifacts` 等字段。

**示例：合成外推验证**
```powershell
$env:SATEMIS_GROUND_TRUTH__ENABLED = "true"
satemis run --species NO2 --ground-truth
# 在 synthetic 模式下从 truth_column 采样生成伪地面站观测，检验 gap→精度结论的外推性
```

### 2.4 `satemis import-l2` — 手动导入 Sentinel-5P L2 granules

```powershell
satemis import-l2 ./my_downloads/           # 递归扫描 .nc
satemis import-l2 ./my_downloads/ --force   # 跳过 schema 校验强制导入
```

行为：
1. 递归扫描指定目录下所有 `*.nc` 文件
2. 对每个文件调用 `validate_granule()` 做 schema 校验（物种列变量存在）
3. 自动从列变量识别物种（NO2/CO/O3/...），从文件名时间戳或 mtime 推断日期
4. 拷贝到 `<data_root>/s5p/<species>/<SPECIES>_YYYY-MM-DD.nc` 缓存目录
5. 输出每个文件的处理结果 + 总体摘要 JSON

**用途**：从 ESA Copernicus / CDS 门户下载好的 L2 granule 文件可批量注入本地缓存，后续 real 模式的 `satemis run` 会优先读缓存、避免重复下载。

### 2.5 `satemis tune` — 独立 Optuna 调参

```powershell
satemis tune --species NO2 --start 2026-03-01 --end 2026-03-08 --trials 20
```

仅对指定通道执行训练与超参搜索，不运行完整管道；输出最佳参数、验证指标和耗时 JSON。日常 `satemis run` 默认 `N_TRIALS=0`，不会隐式启动 Optuna。`--trials` 必须为正整数。

---

## 三、环境变量配置

所有配置通过 `SATEMIS_*` 环境变量控制（嵌套用 `__`）。完整列表见 `src/satemis/config.py`。

### 3.1 数据源开关

```powershell
$env:SATEMIS_DATA_SOURCE = "synthetic"   # 默认，离线合成源
$env:SATEMIS_DATA_SOURCE = "real"        # 真实 Sentinel-5P + ERA5
```

### 3.2 网格定义

```powershell
$env:SATEMIS_GRID__LAT_MIN = 32.0
$env:SATEMIS_GRID__LAT_MAX = 36.0
$env:SATEMIS_GRID__LON_MIN = 110.0
$env:SATEMIS_GRID__LON_MAX = 114.0
$env:SATEMIS_GRID__RESOLUTION_DEG = 0.2
```

### 3.3 合成源参数（仅 synthetic 模式生效）

```powershell
$env:SATEMIS_SYNTHETIC__N_DAYS = 12
$env:SATEMIS_SYNTHETIC__GAP_FRACTION = 0.35        # 缺测比例（用于缺口率扫描）
$env:SATEMIS_SYNTHETIC__POINT_SOURCE_COUNT = 8
$env:SATEMIS_SYNTHETIC__REGIONAL_SOURCE_COUNT = 3
$env:SATEMIS_SYNTHETIC__NOISE_SIGMA = 0.06
$env:SATEMIS_SYNTHETIC__CLOUD_EVENTS_PER_DAY = 2
```

### 3.4 ML 引擎与贝叶斯后验校正

```powershell
$env:SATEMIS_ML__N_TRIALS = 0               # 默认关闭 Optuna；调参使用 satemis tune
$env:SATEMIS_ML__XGB_N_ESTIMATORS = 400
$env:SATEMIS_ML__KRIGE_ENABLED = "true"     # 是否启用 RK 克里金残差
$env:SATEMIS_ML__EXPLAIN_ENABLED = "true"   # SHAP 可解释性
$env:SATEMIS_ML__BAYESIAN_ENABLED = "true"  # ★ 开启 EMG → 贝叶斯后验校正
```

`BAYESIAN_ENABLED=true` 时，pipeline 在 EMG 点源拟合后追加一层高斯-高斯共轭后验更新：
- 先验：ML 引擎的 `emission_rate` 场 + 10%/90% 分位数区间
- 似然：EMG 拟合出的点源强度 `q`（σ 由 R² 调整）
- 后验：仅覆盖 EMG 验证过的源簇 cell，输出 `posterior_weight` 变量标注 EMG 影响份额
- 用途：**降低点源通量估计对风速场扰动的敏感性**（T-2）

### 3.5 地面站交叉比对

```powershell
$env:SATEMIS_GROUND_TRUTH__ENABLED = "true"
$env:SATEMIS_GROUND_TRUTH__NETWORKS = '["tccon","pandora"]'   # 留空 = 全部内置网络
```

开启后 pipeline 在重构完成后自动：
1. 从 `builtin_stations()`（12 TCCON + 1 Pandora 经来源核对的内置站点）筛选落在目标网格内的站点
2. **synthetic 模式**：从 `truth_column` 最近邻采样生成伪观测
3. **real 模式**：读 `<data_root>/ground/<network>/<station_id>.nc`（CDK/Pandora 格式，best-effort）
4. 调用 `cross_validate()` 在每站比较重构值 vs 观测值
5. 输出每站 `R² / bias / RMSE / 相对偏差` + 整体聚合 → 存到 `runs/{id}/ground_truth_report.json`

**接入真实地面站数据**：把 TCCON CDK / Pandora 的 NetCDF 文件放到 `<data_root>/ground/<network>/<station_id>.nc`，station_id 用 `stations_catalog.yaml` 里的 id（如 `tccon_zp` / `pand_bj_radi`）。

### 3.6 真实 Sentinel-5P 凭证（仅 real 模式）

参考 `.env.example`，主要是 CDS API token 与 CDS OData 凭证。

---

## 四、API 端点速查

启动 `uvicorn satemis.api.main:app --port 8000` 后：

| 端点 | 方法 | 说明 |
|---|---|---|
| `/api/v1/auth/token` | POST | OAuth2 password flow 登录，拿 access_token |
| `/api/v1/jobs` | POST | 提交后台 pipeline 任务 |
| `/api/v1/jobs/{id}` | GET | 查询任务状态 |
| `/api/v1/jobs/{id}/events` | GET | SSE 实时日志流 |
| `/api/v1/runs` | GET | 列出所有 pipeline 运行（按时间倒序） |
| `/api/v1/runs/{id}/grid` | GET | 稀疏网格数据（地图渲染用） |
| `/api/v1/runs/{id}/timeseries` | GET | 单点时间序列 |
| `/api/v1/runs/{id}/report` | GET | 评估报告 JSON |
| `/api/v1/runs/{id}/report.html` | GET | 自包含 HTML 报告 |
| `/api/v1/runs/{id}/audit` | GET | 审查规则触发事件列表 |
| **`/api/v1/runs/{id}/ground-truth`** | GET | **地面站交叉比对报告（需 `--ground-truth` 开启）** |
| `/api/v1/runs/{id}/emission` | GET | GeoJSON 排放场（含 EMG 验证标记） |

交互式文档：`http://localhost:8000/docs`（Swagger UI）。

---

## 五、回归门禁（CI Robustness Gates）

缺口率扫描已固化为 pytest 回归测试，防止后续模型迭代引入隐性退化：

```powershell
# 单独跑回归门禁
.\.venv\Scripts\python.exe -m pytest tests/regression -m regression -v
```

测试逻辑（`tests/regression/test_gap_sweep.py`）：
- 对 `gap_fraction ∈ {0.15, 0.30, 0.45, 0.60, 0.70}` 5 档参数化
- 每档重建 L3 → 主通道 fit/reconstruct → 取 R²
- 按 `MIN_R2_BY_GAP` 字典 per-row gate：
  - gap 0.15–0.45 → R² ≥ 0.80
  - gap 0.60 → R² ≥ 0.75
  - gap 0.70 → R² ≥ 0.60
- 任一档跌破门槛即 CI 失败

`tests/regression/test_wind_sensitivity.py` 进一步对基准风场、风速 +1.5 m/s、风向 ±10° 四种场景执行解析 EMG 合成羽流回归，要求拟合均通过验证且源强变异系数 < 0.15，防止风场扰动导致点源反演失稳。

`pyproject.toml` 已注册 `@pytest.mark.regression` 标记，CI 会独立执行完整回归门禁。

---

## 六、测试与质量闸门

```powershell
# 全量测试（含覆盖率门槛 70%）
.\.venv\Scripts\python.exe -m pytest tests/ -ra -q

# 静态检查
.\.venv\Scripts\python.exe -m ruff check src/satemis/
.\.venv\Scripts\python.exe -m mypy src/satemis/ --ignore-missing-imports

# 只跑单元测试
.\.venv\Scripts\python.exe -m pytest tests/unit -q

# 只跑集成测试
.\.venv\Scripts\python.exe -m pytest tests/integration -q

# 只跑回归门禁
.\.venv\Scripts\python.exe -m pytest tests/regression -m regression -q
```

质量门槛（当前全部达标）：
- 单元测试覆盖率 ≥ 70%
- ruff 0 error
- mypy 0 error
- NO₂ 主通道 R² ≥ 0.80
- CO 通道（缺失率 > 0.5）R² ≥ 0.88
- O₃ 通道 R² ≥ 0.78 且优于分离基线
- 审查引擎注入异常召回率 ≥ 90%
- API P95 < 500ms

---

## 七、目录结构

| 目录 | 说明 |
|---|---|
| `src/satemis/core` | 网格/契约/审查规则/质量分级/孤立性判据/可复现性 |
| `src/satemis/data` | Provider 协议、合成源、真实接入（tropomi/era5/dem/geoschem）、**ground_stations**、StorageBackend |
| `src/satemis/channels` | 主通道 / CO 通道 / O₃ 通道（统一 BaseChannel + 注册表） |
| `src/satemis/inversion` | ML 反演引擎、EMG 点源拟合（含风场约束）、**bayesian 后验校正**、清单比对 |
| `src/satemis/validation` | RF 独立验证、**ground_truth 交叉比对**、指标计算 |
| `src/satemis/services` | 编排单一真源（API / Celery / Prefect 共用） |
| `src/satemis/api` | FastAPI 路由（auth/jobs/runs）+ RBAC |
| `src/satemis/db` | SQLAlchemy 模型 + session |
| `src/satemis/workers` | Celery 异步任务 |
| `src/satemis/pipelines` | Prefect 流程（daily / backfill） |
| `frontend` | React 18 + TS + deck.gl + ECharts |
| `tests/unit` | 单元测试（站点、EMG、Bayesian、ground_truth 等） |
| `tests/integration` | 集成测试（pipeline 端到端、API 路由） |
| `tests/regression` | CI 回归门禁（gap sweep） |

---

## 八、典型工作流

### 8.1 离线开发 / 演示
```powershell
satemis run --species NO2 --ground-truth
# 5 秒级跑完，输出含 ground_truth 字段
```

### 8.2 真实 S-5P + 地面站外推验证
```powershell
# 1) 配置 .env（CDS 凭证）
# 2) 放地面站数据到 <data_root>/ground/tccon/*.nc
# 3) 切换数据源并跑管道
$env:SATEMIS_DATA_SOURCE = "real"
$env:SATEMIS_GROUND_TRUTH__ENABLED = "true"
satemis run --species NO2 --start 2025-06-01 --end 2025-06-08 --ground-truth
# 输出 ground_truth.overall_r2 即真实 TCCON 站点上的重构精度 → 检验合成基准结论的外推性
```

### 8.3 批量导入已下载的 L2 granules
```powershell
satemis import-l2 D:\downloads\s5p_2025_06\
# 缓存注入后，下一次 real 模式 run 会命中本地缓存
```

### 8.4 开启贝叶斯后验校正
```powershell
$env:SATEMIS_ML__BAYESIAN_ENABLED = "true"
satemis run --species NO2
# 输出 bayesian.posterior_shift_mean / emg_weight_mean 等字段
```
