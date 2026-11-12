from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "clean_model_dataset_v5_shanghai_augmented.xlsx"
OUT = ROOT / "outputs" / "paper1_taskA_matched_horizon_baselines"
TAB, FIG = OUT / "tables", OUT / "figures"
for p in (OUT, TAB, FIG): p.mkdir(parents=True, exist_ok=True)
TEST_START, TEST_END = "2023-01", "2025-12"
HORIZONS = [1, 3, 6, 12]


def load():
    d = pd.read_excel(DATA, sheet_name="Clean_Main_2012_2025")
    d.period = d.period.astype(str); d = d.sort_values("period").drop_duplicates("period")
    d["y"] = pd.to_numeric(d.shanghai_teu_10k, errors="coerce")
    d["month"] = d.period.str[5:7].astype(int)
    d = d[d.y.notna()].copy()
    return d.reset_index(drop=True)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float); den = np.abs(y) + np.abs(p)
    return {"MAE": float(np.mean(np.abs(y-p))), "RMSE": float(np.sqrt(np.mean((y-p)**2))),
            "MAPE_%": float(np.mean(np.abs((y-p)/y))*100), "sMAPE_%": float(np.mean(np.where(den == 0, 0, 2*np.abs(y-p)/den))*100),
            "R2": float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2))}


def seasonal_naive(y, h):
    x = list(np.asarray(y, float)); out=[]
    for _ in range(h): out.append(x[-12] if len(x) >= 12 else x[-1]); x.append(out[-1])
    return np.asarray(out)


def ets(y, h):
    fit = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=12, initialization_method="estimated").fit(optimized=True)
    return np.asarray(fit.forecast(h), float)


def ridge_recursive(y, h, alpha=10.0):
    y = np.asarray(y, float); lags = [1, 2, 3, 6, 12]
    rows, targets = [], []
    for i in range(12, len(y)):
        rows.append([y[i-k] for k in lags] + [np.sin(2*np.pi*((i % 12)+1)/12), np.cos(2*np.pi*((i % 12)+1)/12)])
        targets.append(y[i])
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(np.asarray(rows), np.asarray(targets))
    x = list(y); out=[]
    for i in range(h):
        idx = len(x); m = (idx % 12) + 1
        row = [[x[idx-k] for k in lags] + [np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12)]]
        val = float(model.predict(row)[0]); out.append(val); x.append(val)
    return np.asarray(out)


def main():
    d = load(); periods = d.period.tolist(); records=[]
    first = periods.index(TEST_START); last = periods.index(TEST_END)
    for h in HORIZONS:
        # Start early enough that the first test month also has a valid
        # h-step-ahead forecast. Origins before the test period use no test
        # observations; only their targets inside the test period are scored.
        for oi in range(max(12, first-h), last+1):
            if oi < 12 or oi + h > last + 1: continue
            origin, train = periods[oi], d.iloc[:oi+1]
            future = d.iloc[oi+1:oi+1+h]
            forecasts = {"SeasonalNaive": seasonal_naive(train.y, h), "ETS": ets(train.y.to_numpy(), h), "Ridge": ridge_recursive(train.y.to_numpy(), h)}
            for model, pred in forecasts.items():
                for j, (_, row) in enumerate(future.iterrows(), 1):
                    if row.period < TEST_START or row.period > TEST_END: continue
                    records.append({"origin": origin, "period": row.period, "horizon": h, "step": j, "model": model, "actual": float(row.y), "predicted": float(pred[j-1])})
    detail = pd.DataFrame(records)
    detail.to_csv(TAB / "taskA_matched_horizon_predictions.csv", index=False, encoding="utf-8-sig")
    rows=[]
    for (h, model), g in detail.groupby(["horizon", "model"]): rows.append({"horizon": h, "model": model, "n_pairs": len(g), **metrics(g.actual, g.predicted)})
    result = pd.DataFrame(rows).sort_values(["horizon", "RMSE"]); result.to_csv(TAB / "taskA_matched_horizon_metrics.csv", index=False, encoding="utf-8-sig")
    # For each horizon, plot the first available forecast path and actual observations.
    for h in HORIZONS:
        g = detail[(detail.horizon == h) & (detail.origin == periods[first-1])].sort_values("period")
        if g.empty: continue
        fig, ax = plt.subplots(figsize=(10, 4.2)); actual = g.drop_duplicates("period"); ax.plot(actual.period, actual.actual, color="black", lw=2, label="Actual")
        for model, color in [("SeasonalNaive", "#9C755F"), ("ETS", "#4E79A7"), ("Ridge", "#59A14F")]:
            q = g[g.model == model]; ax.plot(q.period, q.predicted, lw=1.4, label=model, color=color)
        ax.set_title(f"Task A matched-horizon forecast, H={h}", loc="left", fontweight="bold"); ax.set_ylabel("10k TEU"); ax.tick_params(axis="x", rotation=60); ax.legend(ncol=4); fig.tight_layout(); fig.savefig(FIG / f"fig_taskA_H{h}.png", dpi=400); fig.savefig(FIG / f"fig_taskA_H{h}.svg"); plt.close(fig)
    manifest = {"version": "Task A baseline", "target": "shanghai_teu_10k", "training": "expanding window through each origin", "test": f"{TEST_START} to {TEST_END}", "horizons": HORIZONS, "models": ["SeasonalNaive", "ETS", "Ridge"], "external_variables": "none; target-history-only baseline", "leakage_control": "each forecast uses only values through its origin; recursive Ridge uses its own prior predictions", "aggregation": "all valid origin-target pairs within the test period"}
    (OUT / "taskA_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# Task A: Matched-horizon rolling-origin baselines", "", "The experiment evaluates H=1, 3, 6 and 12 month forecasts using expanding-window rolling origins. Every origin uses only observations available up to that origin. This is a literature-aligned benchmark and is separate from the static 36-month planning stress test.", "", "## Metrics", "", result.round(4).to_markdown(index=False), "", "## Files", "", f"- Metrics: `{TAB / 'taskA_matched_horizon_metrics.csv'}`", f"- Predictions: `{TAB / 'taskA_matched_horizon_predictions.csv'}`", f"- Figures: `{FIG}`", f"- Manifest: `{OUT / 'taskA_manifest.json'}`"]
    (OUT / "taskA_report.md").write_text("\n".join(report), encoding="utf-8")
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__": main()
