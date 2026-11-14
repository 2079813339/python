"""福建区域臭氧 MDA8 基准实验 —— 全局配置。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
REPORTS = OUTPUTS / "reports"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"

for _p in (DATA_INTERIM, DATA_PROCESSED, REPORTS, TABLES, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# 福建省 9 个地级市（与 CNEMC 站点信息表中的城市名一致）
FJ_CITIES = ["福州", "厦门", "泉州", "漳州", "莆田", "三明", "南平", "龙岩", "宁德"]

STUDY_YEARS = [2019, 2020, 2021, 2022]
TRAIN_YEARS = [2019, 2020]
VAL_YEAR = 2021
TEST_YEAR = 2022

# ---- MDA8 计算规则（北京时间 UTC+8，日内 8 小时滑动窗口）----
# HJ 633-2012 / HJ 663-2013：8 小时滑动均值至少需要 6 个有效小时；
# 日均值达标评价另要求每日有效窗口数，这里采用“至少 6 个有效窗口”作为日内 MDA8 有效条件，
# 并同时输出严格阈值（>=13 个有效窗口）下的覆盖情况作为敏感性说明。
MIN_HOURS_PER_WINDOW = 6
MIN_WINDOWS_PER_DAY = 6
MIN_WINDOWS_PER_DAY_STRICT = 13

# 逐小时 O3 的物理合理范围（μg/m³），超出视为异常值并置缺测
O3_VALID_RANGE = (0.0, 1000.0)

STATION_INFO_CSV = DATA_RAW / "Station_information.csv"
OZONE_ZIP = DATA_RAW / "Hourly_surface_ozone.zip"
QUOTSOFT_DIR = DATA_RAW / "quotsoft"
METEO_DIR = DATA_RAW / "openmeteo"

POLLUTANT_TYPES = ["NO2", "CO", "PM2.5", "PM10", "SO2", "O3"]
POLLUTANT_UNITS = {"NO2": "μg/m³", "CO": "mg/m³", "PM2.5": "μg/m³", "PM10": "μg/m³", "SO2": "μg/m³", "O3": "μg/m³"}
