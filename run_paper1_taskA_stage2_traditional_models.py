from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "clean_model_dataset_v5_shanghai_augmented.xlsx"
SRC = ROOT / "outputs" / "paper1_taskA_matched_horizon_baselines" / "tables" / "taskA_matched_horizon_predictions.csv"
OUT = ROOT / "outputs" / "paper1_taskA_stage2_traditional_models"
TAB, FIG = OUT / "tables", OUT / "figures"
for p in (OUT, TAB, FIG): p.mkdir(parents=True, exist_ok=True)
TEST_START, TEST_END = "2023-01", "2025-12"
HORIZONS = [1, 3, 6, 12]
INTERNAL_ORIGINS = ["2018-12", "2019-12", "2020-12", "2021-12"]
LAGS = [1, 2, 3, 6, 12]


def load():
    d = pd.read_excel(DATA, sheet_name="Clean_Main_2012_2025")
    d.period = d.period.astype(str); d = d.sort_values("period").drop_duplicates("period")
    d["y"] = pd.to_numeric(d.shanghai_teu_10k, errors="coerce"); d["month"] = d.period.str[5:7].astype(int)
    return d[d.y.notna()].reset_index(drop=True)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float); den = np.abs(y) + np.abs(p)
    return {"MAE": float(np.mean(np.abs(y-p))), "RMSE": float(np.sqrt(np.mean((y-p)**2))), "MAPE_%": float(np.mean(np.abs((y-p)/y))*100), "sMAPE_%": float(np.mean(np.where(den == 0, 0, 2*np.abs(y-p)/den))*100), "R2": float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2))}


def features(y, month, h):
    y = np.asarray(y, float); month = np.asarray(month, int); X, Y = [], []
    for i in range(12, len(y)-h+1):
        X.append([y[i-k] for k in LAGS] + [np.sin(2*np.pi*month[i]/12), np.cos(2*np.pi*month[i]/12)])
        Y.append(y[i:i+h])
    return np.asarray(X), np.asarray(Y)


def origin_x(y, month):
    i = len(y); m = int(month[-1])
    return np.asarray([[y[i-k] for k in LAGS] + [np.sin(2*np.pi*((m % 12)+1)/12), np.cos(2*np.pi*((m % 12)+1)/12)]])


def ml_forecast(y, month, h, name, params):
    X, Y = features(y, month, h)
    if name == "ElasticNet":
        base = ElasticNet(alpha=float(params["alpha"]), l1_ratio=float(params["l1_ratio"]), max_iter=30000, random_state=42)
    else:
        base = SVR(kernel="rbf", C=float(params["C"]), epsilon=float(params["epsilon"]), gamma=float(params["gamma"]))
    model = make_pipeline(StandardScaler(), MultiOutputRegressor(base)).fit(X, Y)
    return np.maximum(np.asarray(model.predict(origin_x(y, month))[0], float), 0.0)


def sarima_forecast(y, h, params):
    fit = SARIMAX(y, order=tuple(params["order"]), seasonal_order=tuple(params["seasonal_order"]), trend="c", enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=300)
    return np.maximum(np.asarray(fit.forecast(h), float), 0.0)


def candidate_grid(name):
    if name == "SARIMA":
        return [{"order": list(o), "seasonal_order": list(s)} for o in [(0,1,1),(1,1,0),(1,1,1),(2,1,0)] for s in [(0,1,1,12),(1,1,0,12)]]
    if name == "ElasticNet":
        return [{"alpha": a, "l1_ratio": r} for a in [0.01, 0.1, 1.0, 10.0] for r in [0.1, 0.5, 0.9]]
    return [{"C": c, "epsilon": e, "gamma": g} for c in [0.1, 1.0, 10.0] for e in [0.1, 1.0] for g in ["scale", 0.01]]


def forecast(name, y, month, h, params):
    return sarima_forecast(y, h, params) if name == "SARIMA" else ml_forecast(y, month, h, name, params)


def tune(d, name, h):
    rows = []
    for params in candidate_grid(name):
        fold_metrics = []
        for origin in INTERNAL_ORIGINS:
            ix = d.index[d.period == origin]
            if len(ix) == 0: continue
            oi = int(ix[0]); future = d.iloc[oi+1:oi+1+h]
            if len(future) < h: continue
            try:
                pred = forecast(name, d.y.iloc[:oi+1].to_numpy(), d.month.iloc[:oi+1].to_numpy(), h, params)
                fold_metrics.append(metrics(future.y, pred))
            except Exception:
                fold_metrics = []; break
        if not fold_metrics: continue
        rows.append({"model": name, "horizon": h, "params": json.dumps(params, ensure_ascii=False), "mean_RMSE": np.mean([x["RMSE"] for x in fold_metrics]), "mean_MAPE_%": np.mean([x["MAPE_%"] for x in fold_metrics]), "n_folds": len(fold_metrics)})
    cv = pd.DataFrame(rows).sort_values(["mean_RMSE", "mean_MAPE_%"]).reset_index(drop=True)
    return cv.iloc[0].to_dict(), cv


def main():
    d = load(); periods = d.period.tolist(); first, last = periods.index(TEST_START), periods.index(TEST_END)
    selected, cv_all = [], []
    for name in ["SARIMA", "ElasticNet", "SVR"]:
        for h in HORIZONS:
            best, cv = tune(d, name, h); selected.append(best); cv_all.append(cv)
    selected_df = pd.DataFrame(selected); cv_df = pd.concat(cv_all, ignore_index=True)
    selected_df.to_csv(TAB / "taskA_stage2_selected_hyperparameters.csv", index=False, encoding="utf-8-sig")
    cv_df.to_csv(TAB / "taskA_stage2_internal_cv_selection_records.csv", index=False, encoding="utf-8-sig")
    params_map = {(r.model, int(r.horizon)): json.loads(r.params) for _, r in selected_df.iterrows()}
    records = []
    # Test origins begin early enough to score the first test month for each H.
    for h in HORIZONS:
        for oi in range(max(12, first-h), last+1):
            origin = periods[oi]; future = d.iloc[oi+1:oi+1+h]
            if len(future) < h: continue
            for name in ["SARIMA", "ElasticNet", "SVR"]:
                try: pred = forecast(name, d.y.iloc[:oi+1].to_numpy(), d.month.iloc[:oi+1].to_numpy(), h, params_map[(name,h)])
                except Exception: pred = np.repeat(d.y.iloc[oi], h)
                for j, (_, row) in enumerate(future.iterrows(), 1):
                    if TEST_START <= row.period <= TEST_END: records.append({"origin": origin, "period": row.period, "horizon": h, "step": j, "model": name, "actual": float(row.y), "predicted": float(pred[j-1])})
    pred = pd.DataFrame(records)
    # Add original ETS baseline rows for directly comparable block and lead tables.
    base = pd.read_csv(SRC); base.period = base.period.astype(str)
    base = base.rename(columns={"predicted": "predicted"})[["origin","period","horizon","step","model","actual","predicted"]]
    all_pred = pd.concat([base, pred], ignore_index=True).sort_values(["horizon","model","origin","step"])
    all_pred.to_csv(TAB / "taskA_stage2_predictions.csv", index=False, encoding="utf-8-sig")
    lead_rows, block_rows, step_rows = [], [], []
    for h in HORIZONS:
        for model, g in all_pred[(all_pred.horizon == h) & (all_pred.step == h)].groupby("model"):
            lead_rows.append({"lead": h, "model": model, "n": len(g), **metrics(g.actual, g.predicted)})
        for model, g in all_pred[all_pred.horizon == h].groupby("model"):
            block_rows.append({"block_horizon": h, "model": model, "n": len(g), **metrics(g.actual, g.predicted)})
        for step in range(1, h+1):
            for model, g in all_pred[(all_pred.horizon == h) & (all_pred.step == step)].groupby("model"):
                step_rows.append({"block_horizon": h, "step": step, "model": model, "n": len(g), **metrics(g.actual, g.predicted)})
    lead_df, block_df, step_df = pd.DataFrame(lead_rows), pd.DataFrame(block_rows), pd.DataFrame(step_rows)
    lead_df.to_csv(TAB / "taskA_stage2_lead_specific_metrics.csv", index=False, encoding="utf-8-sig")
    block_df.to_csv(TAB / "taskA_stage2_block_metrics.csv", index=False, encoding="utf-8-sig")
    step_df.to_csv(TAB / "taskA_stage2_stepwise_metrics.csv", index=False, encoding="utf-8-sig")
    # Improvement relative to ETS at each lead.
    ets = lead_df[lead_df.model == "ETS"].set_index("lead")
    imp = lead_df.copy(); imp["RMSE_improvement_vs_ETS_%"] = imp.apply(lambda r: (ets.loc[r.lead,"RMSE"]-r.RMSE)/ets.loc[r.lead,"RMSE"]*100, axis=1); imp["MAPE_improvement_vs_ETS_pp"] = imp.apply(lambda r: ets.loc[r.lead,"MAPE_%"]-r["MAPE_%"], axis=1)
    imp.to_csv(TAB / "taskA_stage2_improvement_vs_ETS.csv", index=False, encoding="utf-8-sig")
    mean = lead_df.groupby("model").agg(MeanLead_MAE=("MAE","mean"), MeanLead_RMSE=("RMSE","mean"), MeanLead_MAPE=("MAPE_%","mean"), MeanLead_sMAPE=("sMAPE_%","mean"), MeanLead_R2=("R2","mean")).reset_index(); mean.to_csv(TAB / "taskA_stage2_mean_lead_summary.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7.5,4.4))
    for model, color in [("ETS","#4E79A7"),("SARIMA","#F28E2B"),("ElasticNet","#59A14F"),("SVR","#E15759")]:
        q = lead_df[lead_df.model == model].sort_values("lead")
        if not q.empty: ax.plot(q.lead, q.RMSE, marker="o", lw=1.5, label=model, color=color)
    ax.set_xlabel("Lead (months)"); ax.set_ylabel("RMSE (10k TEU)"); ax.set_title("Task A Stage 2 lead-specific RMSE", loc="left", fontweight="bold"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "fig_taskA_stage2_lead_rmse.png", dpi=400); fig.savefig(FIG / "fig_taskA_stage2_lead_rmse.svg"); plt.close(fig)
    manifest = {"version":"Task A Stage 2", "training":"expanding window through each forecast origin", "internal_tuning_origins": INTERNAL_ORIGINS, "test":f"{TEST_START} to {TEST_END}", "horizons":HORIZONS, "models":["ETS baseline from Stage 1","SARIMA","ElasticNet","SVR"], "external_variables":"none; target-history-only", "lead_definition":"horizon=lead and step=lead", "block_definition":"all steps 1:H pooled", "leakage_control":"hyperparameters selected only with origins no later than 2021-12; test values are never used for selection"}
    (OUT / "taskA_stage2_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# Task A Stage 2: Traditional and machine-learning baselines", "", "This experiment keeps the frozen Task A protocol and adds low-order SARIMA, ElasticNet and RBF-SVR. Each horizon has its own model and hyperparameters. Hyperparameters are selected using internal origins through 2021-12 and then frozen for the 2023-2025 expanding-origin test.", "", "## Lead-specific metrics", "", lead_df.sort_values(["lead","RMSE"]).round(4).to_markdown(index=False), "", "## Mean across leads", "", mean.round(4).to_markdown(index=False), "", "## Interpretation", "", "The lead-specific table is the primary Task A comparison. Block metrics and step-wise metrics are supplementary views and should not be described as fixed-lead accuracy.", "", "## Files", "", f"- Lead metrics: `{TAB / 'taskA_stage2_lead_specific_metrics.csv'}`", f"- Block metrics: `{TAB / 'taskA_stage2_block_metrics.csv'}`", f"- Step-wise metrics: `{TAB / 'taskA_stage2_stepwise_metrics.csv'}`", f"- Predictions: `{TAB / 'taskA_stage2_predictions.csv'}`", f"- Hyperparameters: `{TAB / 'taskA_stage2_selected_hyperparameters.csv'}`", f"- CV records: `{TAB / 'taskA_stage2_internal_cv_selection_records.csv'}`", f"- ETS improvement: `{TAB / 'taskA_stage2_improvement_vs_ETS.csv'}`", f"- Figure: `{FIG / 'fig_taskA_stage2_lead_rmse.png'}`", f"- Manifest: `{OUT / 'taskA_stage2_manifest.json'}`"]
    (OUT / "taskA_stage2_report.md").write_text("\n".join(report), encoding="utf-8")
    print(lead_df.sort_values(["lead","RMSE"]).to_string(index=False), "\nMEAN\n", mean.to_string(index=False), flush=True)


if __name__ == "__main__": main()
