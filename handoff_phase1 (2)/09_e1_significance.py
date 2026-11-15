"""步骤 9：E1 增益的配对显著性检验。

对同一批 2022 站点日做配对自助法（2000 次重采样），给出 ΔRMSE 的 95% 置信区间，
用于区分“真实增益”与“随机波动”；同时按城市统计留城市折的改善方向一致性。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fjozone import config  # noqa: E402

VERSIONS = ["E1-0", "E1-1", "E1-2", "E1-3"]
N_BOOT = 2000
SEED = 42


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main() -> None:
    preds = {}
    for v in VERSIONS:
        p = pd.read_csv(config.TABLES / f"prediction_2022_{v}_B2_XGB.csv", parse_dates=["date"])
        preds[v] = p.set_index(["station_code", "date"]).sort_index()

    base = preds["E1-0"]
    idx = base.index
    y = base["y_true"].to_numpy()

    rows, boot_rows = [], []
    rng = np.random.default_rng(SEED)
    for v in VERSIONS[1:]:
        p = preds[v].reindex(idx)
        assert p["y_true"].notna().all(), f"{v} 与 E1-0 样本未对齐"
        yv = p["y_true"].to_numpy()
        pv = p["y_pred"].to_numpy()
        thr = np.nanpercentile(yv, 95)
        m = {}
        for name, yy, pp in (("all", yv, pv), ("p95", yv[yv >= thr], pv[yv >= thr])):
            m[f"RMSE_{name}"] = _rmse(yy, pp)
            m[f"MAE_{name}"] = float(np.mean(np.abs(yy - pp)))
        m["RMSE_all_base"] = _rmse(y, base["y_pred"].to_numpy())
        m["RMSE_p95_base"] = _rmse(y[y >= thr], base["y_pred"].to_numpy()[y >= thr])
        rows.append({"version": v, **m})

        # 配对自助法：同一重采样索引同时评估两个版本
        n = len(yv)
        di = np.empty(N_BOOT)
        dp = np.empty(N_BOOT)
        for b in range(N_BOOT):
            s = rng.integers(0, n, n)
            di[b] = _rmse(yv[s], pv[s]) - _rmse(y[s], base["y_pred"].to_numpy()[s])
            sel = yv[s] >= thr
            dp[b] = _rmse(yv[s][sel], pv[s][sel]) - _rmse(y[s][sel], base["y_pred"].to_numpy()[s][sel])
        boot_rows.append({
            "version": v, "dRMSE": di.mean(),
            "dRMSE_ci_lo": np.percentile(di, 2.5), "dRMSE_ci_hi": np.percentile(di, 97.5),
            "p_dRMSE_lt0": float((di < 0).mean()),
            "dRMSE_p95": dp.mean(),
            "dRMSE_p95_ci_lo": np.percentile(dp, 2.5), "dRMSE_p95_ci_hi": np.percentile(dp, 97.5),
        })

    sig = pd.DataFrame(rows)
    boot = pd.DataFrame(boot_rows)
    sig.to_csv(config.TABLES / "e1_lag_paired_metrics.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(config.TABLES / "e1_lag_bootstrap.csv", index=False, encoding="utf-8-sig")

    city = pd.read_csv(config.TABLES / "e1_lag_spatial_by_city.csv")
    city = city[city["model"] == "B2_XGB"].pivot_table(index="city", columns="version", values="RMSE")
    for v in VERSIONS[1:]:
        city[f"d_{v}"] = city[v] - city["E1-0"]
    city.to_csv(config.TABLES / "e1_lag_spatial_city_delta.csv", encoding="utf-8-sig")

    sign = pd.DataFrame({
        "version": VERSIONS[1:],
        "改善城市数": [int((city[f"d_{v}"] < 0).sum()) for v in VERSIONS[1:]],
        "城市总数": [len(city)] * 3,
    })

    lines = [
        "# E1 滞后特征增益的显著性检验",
        "",
        "对 2022 测试集同一批站点日做配对自助法（2000 次，seed=42）：每次重采样同时评估两个版本，",
        "差值不依赖样本组成，可直接判断增益是否超出随机波动。",
        "",
        "## 2022 独立测试：配对差值 95% 置信区间",
        "",
        boot.round(3).to_markdown(index=False),
        "",
        "`p_dRMSE_lt0` 为自助样本中“该版本 RMSE 更低”的比例，可读作单侧概率。",
        "",
        "## 留城市（9 折）：改善方向一致性（XGB）",
        "",
        sign.to_markdown(index=False),
        "",
        city.round(2).to_markdown(),
        "",
        "## 判读",
        "",
        "- 增益幅度约 2%（E1-1/E1-2）与 1%—2%（E1-3），相当于 0.3—0.4 μg/m³；",
        "  若置信区间跨 0，则不能把该幅度视为稳定增益。",
        "- 增量在不同协议下方向不一致（时间测试偏好 D-1 单日滞后，留城市偏好 7 日滚动），",
        "  说明现有滞后信息的作用主要是局地时间惯性，而非可迁移的空间信息。",
        "",
    ]
    (config.REPORTS / "e1_lag_significance.md").write_text("\n".join(lines), encoding="utf-8")
    print(boot.round(3).to_string(index=False))
    print(sign.to_string(index=False))
    print("\n输出: outputs/reports/e1_lag_significance.md")


if __name__ == "__main__":
    main()
