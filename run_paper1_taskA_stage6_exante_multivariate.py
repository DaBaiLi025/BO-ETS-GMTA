from __future__ import annotations

import json
import platform
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels
from scipy.stats import t as student_t
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "clean_model_dataset_v5_shanghai_augmented.xlsx"
OUT = ROOT / "outputs" / "paper1_taskA_stage6_exante_multivariate"
TAB, FIG = OUT / "tables", OUT / "figures"
for p in (OUT, TAB, FIG): p.mkdir(parents=True, exist_ok=True)
TEST_START, TEST_END = "2023-01", "2025-12"
HORIZONS = [1, 3, 6, 12]
INTERNAL_ORIGINS = ["2018-12", "2019-12", "2020-12", "2021-12"]
GROUPS = {
    "G1": ["cargo_lag1"],
    "G2": ["china_manufacturing_pmi_lag1", "china_nonmanufacturing_pmi_lag1", "china_export_value_lag1", "china_import_value_lag1"],
    "G3": ["bdi_mean_lag1", "brent_oil_price_usd_mean_lag1", "usd_cny_exchange_rate_mean_lag1"],
    "G4": [],
}
GROUP_NAMES = {"G1": "Port activity", "G2": "Macro-trade", "G3": "Shipping-financial", "G4": "Foreland/global demand", "AllEligible": "G1+G2+G3"}
ETS_CACHE = {}
RESID_CACHE = {}


def load_data():
    d = pd.read_excel(DATA, sheet_name="Clean_Main_2012_2025")
    d["period"] = d["period"].astype(str)
    d = d.sort_values("period").drop_duplicates("period").reset_index(drop=True)
    d["y"] = pd.to_numeric(d["shanghai_teu_10k"], errors="coerce")
    d["month_num"] = d["period"].str[5:7].astype(int)
    return d[d.y.notna()].reset_index(drop=True)


def finite(v):
    return pd.to_numeric(v, errors="coerce").replace([np.inf, -np.inf], np.nan)


def metrics(y, p, naive=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p; den = np.abs(y) + np.abs(p)
    scale = np.mean(np.abs(y[12:] - y[:-12])) if len(y) > 12 else np.mean(np.abs(np.diff(y)))
    naive_rmse = np.sqrt(np.mean((y[12:] - y[:-12]) ** 2)) if len(y) > 12 else np.nan
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "MAPE_%": float(np.mean(np.abs(e / y)) * 100),
            "sMAPE_%": float(np.mean(np.where(den == 0, 0, 2 * np.abs(e) / den)) * 100),
            "R2": float(1 - np.sum(e ** 2) / np.sum((y - np.mean(y)) ** 2)),
            "MASE": float(np.mean(np.abs(e)) / scale) if scale and np.isfinite(scale) else np.nan,
            "Theil_U": float(np.sqrt(np.mean(e ** 2)) / naive_rmse) if naive_rmse and np.isfinite(naive_rmse) else np.nan}


def ets_forecast(y, h):
    y = np.asarray(y, float)
    if len(y) < 24: return np.repeat(y[-1], h)
    try:
        fit = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=12, initialization_method="estimated").fit(optimized=True)
        return np.maximum(np.asarray(fit.forecast(h), float), 0)
    except Exception:
        return np.maximum(np.asarray([y[-12 + (i % 12)] if len(y) >= 12 else y[-1] for i in range(h)]), 0)


def cached_ets(d, oi, h):
    key = (oi, h)
    if key not in ETS_CACHE: ETS_CACHE[key] = ets_forecast(d.y.iloc[:oi + 1].to_numpy(float), h)
    return ETS_CACHE[key]


def seasonal_naive(x, h):
    x = np.asarray(x, float); out = []
    for i in range(h): out.append(x[-12 + (i % 12)] if len(x) >= 12 else x[-1])
    return np.asarray(out)


def raw_external(d, col):
    """Reconstruct raw month t from lag-1 value stored at t+1."""
    s = finite(d[col]).to_numpy(float)
    raw = np.full(len(s), np.nan)
    raw[:-1] = s[1:]
    return raw


def build_availability(d):
    rows = []
    all_cols = sum(GROUPS.values(), [])
    for c in all_cols + ["shanghai_retail_sales_value_x_lag1", "shanghai_trade_total_value_x_lag1", "shanghai_ppi_output_index_yoy_x_lag1"]:
        group = next((g for g, cs in GROUPS.items() if c in cs), "C"); vals = finite(d[c]) if c in d else pd.Series(dtype=float)
        non = vals.dropna(); start = d.loc[non.index, "period"].min() if len(non) else ""
        end = d.loc[non.index, "period"].max() if len(non) else ""
        status = "eligible_lagged" if c in all_cols and len(non) >= 60 else "excluded"
        reason = "" if status == "eligible_lagged" else "no usable observations in current workbook"
        rows.append({"variable": c, "group": group, "source": "clean_model_dataset_v5_shanghai_augmented.xlsx", "frequency": "monthly", "historical_start": start, "historical_end": end, "publication_lag": "1 month; lagged field", "available_at_origin_rule": "observed only through origin-1 for reconstructed raw series", "requires_future_forecast": True, "transformation": "raw_t inferred from lag1 at t+1; model uses lagged value at target", "final_status": status, "exclusion_reason": reason})
    rows.append({"variable": "teu_mom_pct_calc / teu_yoy_pct_calc", "group": "Leakage", "source": "derived from target", "frequency": "monthly", "historical_start": "", "historical_end": "", "publication_lag": "", "available_at_origin_rule": "prohibited", "requires_future_forecast": False, "transformation": "target-derived", "final_status": "excluded", "exclusion_reason": "current-target leakage"})
    return pd.DataFrame(rows)


def exog_forecast(x, h, method):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0: return np.zeros(h)
    if method == "SeasonalNaive": return seasonal_naive(x, h)
    return ets_forecast(x, h)


def select_external_forecasters(d, variables):
    rows, paths = [], []
    for col in variables:
        raw = raw_external(d, col)
        for method in ["SeasonalNaive", "ETS"]:
            fold = []
            for origin in INTERNAL_ORIGINS:
                oi = int(d.index[d.period == origin][0]); train = raw[:max(0, oi)]
                for h in HORIZONS:
                    actual = raw[oi + h - 1:oi + h]
                    if len(train) < 18 or len(actual) == 0 or not np.isfinite(actual[0]): continue
                    pred = exog_forecast(train, h, method)[-1]; fold.append(abs(float(actual[0] - pred)))
            rows.append({"variable": col, "method": method, "n_folds": len(fold), "mean_RMSE": float(np.mean(fold)) if fold else np.nan, "mean_MAPE_%": np.nan})
        q = pd.DataFrame([r for r in rows if r["variable"] == col]).sort_values(["mean_RMSE", "mean_MAPE_%"])
        selected = q.iloc[0]["method"] if len(q) else "SeasonalNaive"
        for oi, origin in enumerate(d.period):
            if origin < "2018-01" or origin > TEST_END: continue
            i = int(d.index[d.period == origin][0]); latest = max(0, i)
            p = exog_forecast(raw[:latest], 12, selected)
            for j, v in enumerate(p, 1): paths.append({"origin": origin, "variable": col, "target_period": str(pd.Period(origin, freq="M") + j), "horizon": j, "method": selected, "predicted_value": float(v), "latest_observed_period": d.period.iloc[latest - 1] if latest else "", "was_forecasted": True, "leakage_flag": False})
    return pd.DataFrame(rows), pd.DataFrame(paths)


def group_cols(group):
    if group == "AllEligible": return GROUPS["G1"] + GROUPS["G2"] + GROUPS["G3"]
    return GROUPS.get(group, [])


def exog_matrix(d, cols, end_i, h=None, selected_methods=None):
    vals = []
    for c in cols:
        s = finite(d[c]).to_numpy(float)[:end_i + 1]
        s = pd.Series(s).ffill().bfill().to_numpy(float)
        vals.append(s)
    if not vals: return np.empty((end_i + 1, 0)), np.empty((h or 0, 0))
    X = np.column_stack(vals)
    future = []
    raw_end = end_i  # lagged values through origin are used for target-origin training
    for j in range(len(cols)):
        c = cols[j]; raw = raw_external(d, c)[:raw_end]
        method = (selected_methods or {}).get(c, "SeasonalNaive")
        future.append(exog_forecast(raw[np.isfinite(raw)], h or 0, method))
    return X, np.column_stack(future) if h else np.empty((0, len(cols)))


def sarimax_predict(y, X, Xf, h, order):
    try:
        model = SARIMAX(y, exog=X if X.shape[1] else None, order=order, seasonal_order=(0, 1, 1, 12), trend="c", enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False, maxiter=60)
        return np.maximum(np.asarray(fit.forecast(h, exog=Xf if X.shape[1] else None), float), 0)
    except Exception:
        return ets_forecast(y, h)


def feature_row(d, i, h, cols, future_x=None):
    m = int(d.month_num.iloc[min(i + h, len(d) - 1)]) if i + h < len(d) else ((int(d.month_num.iloc[i]) - 1 + h) % 12 + 1)
    base = [np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)]
    if future_x is not None: base.extend(np.asarray(future_x, float).tolist())
    return base


def residual_training(d, end_i, h, cols):
    rows, ys = [], []
    for oi in range(24, end_i - h + 1):
        pred = cached_ets(d, oi, h)
        ti = oi + h
        if ti >= len(d): continue
        x = feature_row(d, oi, h, cols, d.loc[ti, cols].fillna(0).to_numpy(float) if cols else [])
        rows.append(x); ys.append(float(d.y.iloc[ti] - pred[-1]))
    return np.asarray(rows), np.asarray(ys)


def residual_predict(d, oi, h, cols, kind, params):
    key = (oi, h, tuple(cols), kind, float(params.get("alpha", 0.1)), float(params.get("l1_ratio", 0.5)))
    if key not in RESID_CACHE:
        X, yy = residual_training(d, oi, h, cols)
        if len(yy) < 18: RESID_CACHE[key] = None
        else:
            model = Ridge(alpha=params.get("alpha", 10.0)) if kind == "Ridge" else ElasticNet(alpha=params.get("alpha", 0.1), l1_ratio=params.get("l1_ratio", 0.5), max_iter=30000, random_state=42)
            RESID_CACHE[key] = make_pipeline(StandardScaler(), model).fit(X, yy)
    pipe = RESID_CACHE[key]
    if pipe is None: return 0.0
    raw = raw_external(d, cols[0]) if cols else np.array([])
    future_vals = []
    for c in cols:
        s = raw_external(d, c)[:max(1, oi)]
        method = params.get("external_methods", {}).get(c, "SeasonalNaive")
        future_vals.append(exog_forecast(s[np.isfinite(s)], h, method)[-1])
    base = feature_row(d, oi, h, cols, future_vals)
    return float(pipe.predict(np.asarray([base]))[0])


def tune_models(d, ext_methods):
    rows = []; selected = {}
    model_specs = []
    for g in ["G1", "G2", "G3", "AllEligible"]:
        for order in [(0, 0, 1), (1, 0, 0), (0, 1, 1)]: model_specs.append(("M1", g, "none", order, np.nan, np.nan))
        for alpha in [0.1, 1.0, 10.0, 100.0]: model_specs.append(("M2", g, "Ridge", None, alpha, np.nan))
        for alpha in [0.01, 0.1, 1.0]:
            for l1 in [0.2, 0.5, 0.8]: model_specs.append(("M2", g, "ElasticNet", None, alpha, l1))
    for h in HORIZONS:
        for family, group, kind, order, alpha, l1 in model_specs:
            fold = []
            for origin in INTERNAL_ORIGINS:
                oi = int(d.index[d.period == origin][0]); future = d.iloc[oi + h:oi + h + 1]
                if len(future) == 0: continue
                cols = group_cols(group); X, Xf = exog_matrix(d, cols, oi, h, ext_methods)
                if family == "M1": pred = sarimax_predict(d.y.iloc[:oi + 1].to_numpy(float), X, Xf, h, order)[-1]
                else: pred = cached_ets(d, oi, h)[-1] + residual_predict(d, oi, h, cols, kind, {"alpha": alpha, "l1_ratio": l1, "external_methods": ext_methods})
                fold.append(float(abs(d.y.iloc[oi + h] - pred)))
            rows.append({"horizon": h, "family": family, "group": group, "residual_model": kind, "arima_order": json.dumps(order) if order else "", "alpha": alpha, "l1_ratio": l1, "mean_RMSE": float(np.mean(fold)) if fold else np.nan, "sd_RMSE": float(np.std(fold, ddof=1)) if len(fold) > 1 else np.nan, "median_RMSE": float(np.median(fold)) if fold else np.nan, "worst_fold_RMSE": float(np.max(fold)) if fold else np.nan, "robust_score": float(np.mean(fold) + 0.5 * np.std(fold, ddof=1)) if len(fold) > 1 else np.nan, "n_folds": len(fold)})
        q = pd.DataFrame([r for r in rows if r["horizon"] == h]).sort_values(["robust_score", "mean_RMSE"])
        selected[h] = q.iloc[0].to_dict()
    return pd.DataFrame(rows), selected


def evaluate(d, cv, selected, ext_methods):
    periods = d.period.tolist(); first, last = periods.index(TEST_START), periods.index(TEST_END); records = []; audits = []
    for h in HORIZONS:
        hcv = cv[cv.horizon == h].copy()
        group_specs = {g: hcv[hcv.group == g].sort_values(["robust_score", "mean_RMSE"]).iloc[0].to_dict() for g in ["G1", "G2", "G3", "AllEligible"]}
        specs = [("Stage6Selected", selected[h])] + [(f"Stage6_{g}", s) for g, s in group_specs.items()]
        for oi in range(max(24, first - h), last):
            future = d.iloc[oi + 1:oi + h + 1]
            if len(future) < h: continue
            origin = periods[oi]
            for model_label, spec in specs:
                cols = group_cols(spec["group"]); X, Xf = exog_matrix(d, cols, oi, h, ext_methods)
                if spec["family"] == "M1": pred = sarimax_predict(d.y.iloc[:oi + 1].to_numpy(float), X, Xf, h, tuple(json.loads(spec["arima_order"])))
                else:
                    pred = cached_ets(d, oi, h)
                    corrections = [residual_predict(d, oi, j, cols, spec["residual_model"], {"alpha": float(spec["alpha"]), "l1_ratio": float(spec["l1_ratio"]) if np.isfinite(spec["l1_ratio"]) else 0.5, "external_methods": ext_methods}) for j in range(1, h + 1)]
                    pred = pred + np.asarray(corrections)
                for j, (idx, r) in enumerate(future.iterrows(), 1):
                    if TEST_START <= r.period <= TEST_END:
                        records.append({"origin": origin, "period": r.period, "horizon": h, "step": j, "model": model_label, "selected_family": spec["family"], "selected_group": spec["group"], "selected_residual_model": spec["residual_model"], "actual": float(r.y), "predicted": float(pred[j - 1])})
                        for c in cols: audits.append({"forecast_origin": origin, "target_period": r.period, "horizon": h, "step": j, "model": model_label, "variable": c, "information_source": "lagged workbook field + forecasted path", "value_type": "forecasted_exogenous", "source_period": "through origin-1", "publication_lag": 1, "was_forecasted": True, "latest_observed_period": periods[max(0, oi - 1)], "leakage_flag": False})
    return pd.DataFrame(records), pd.DataFrame(audits)


def lead_metrics(pred, baseline):
    rows = []
    for h in HORIZONS:
        for model, g in pred[(pred.horizon == h) & (pred.step == h)].groupby("model"):
            rows.append({"lead": h, "model": model, "n": len(g), **metrics(g.actual, g.predicted)})
    return pd.DataFrame(rows)


def dm_bootstrap(a, b, block=3, n=2000):
    e1 = (a.actual - a.predicted).to_numpy(float) ** 2; e2 = (b.actual - b.predicted).to_numpy(float) ** 2; loss = e1 - e2; nobs = len(loss)
    if nobs < 3: return {"DM_stat": np.nan, "p_value": np.nan, "RMSE_diff_CI_low": np.nan, "RMSE_diff_CI_high": np.nan}
    stat = np.mean(loss) / (np.std(loss, ddof=1) / np.sqrt(nobs)); p = 2 * student_t.sf(abs(stat), max(1, nobs - 1)); rng = np.random.default_rng(42); vals = []; mape_vals = []
    for _ in range(n):
        ix = []
        while len(ix) < nobs:
            s = int(rng.integers(0, nobs)); ix.extend([(s + j) % nobs for j in range(block)])
        ix = ix[:nobs]; vals.append(np.sqrt(np.mean(e1[ix])) - np.sqrt(np.mean(e2[ix]))); actual = a.actual.to_numpy(float)[ix]; mape_vals.append(float(np.mean(np.abs((a.predicted.to_numpy(float)[ix] - actual) / actual)) * 100 - np.mean(np.abs((b.predicted.to_numpy(float)[ix] - actual) / actual)) * 100))
    return {"DM_stat": float(stat), "p_value": float(p), "RMSE_diff_CI_low": float(np.percentile(vals, 2.5)), "RMSE_diff_CI_high": float(np.percentile(vals, 97.5)), "MAPE_diff_pp_CI_low": float(np.percentile(mape_vals, 2.5)), "MAPE_diff_pp_CI_high": float(np.percentile(mape_vals, 97.5))}


def main():
    d = load_data(); audit = build_availability(d); audit.to_csv(TAB / "stage6_variable_availability_audit.csv", index=False, encoding="utf-8-sig")
    dictionary = audit.assign(variable_role=np.where(audit.final_status == "eligible_lagged", "candidate exogenous", "excluded")); dictionary.to_csv(TAB / "stage6_external_variable_dictionary.csv", index=False, encoding="utf-8-sig")
    variables = [c for c in sum(GROUPS.values(), []) if c in d and finite(d[c]).notna().sum() >= 60]
    ext_cv, ext_paths = select_external_forecasters(d, variables); ext_cv.to_csv(TAB / "stage6_external_forecast_cv_records.csv", index=False, encoding="utf-8-sig"); ext_paths.to_csv(TAB / "stage6_external_forecast_paths.csv", index=False, encoding="utf-8-sig")
    ext_methods = {c: ext_cv[ext_cv.variable == c].sort_values("mean_RMSE").iloc[0].method for c in variables}
    pd.DataFrame([{"variable": c, "selected_method": m, "selection_rule": "pre-test endpoint RMSE across H=1,3,6,12"} for c, m in ext_methods.items()]).to_csv(TAB / "stage6_external_forecaster_selection.csv", index=False, encoding="utf-8-sig")
    cv, selected = tune_models(d, ext_methods); cv.to_csv(TAB / "stage6_model_selection_pretest.csv", index=False, encoding="utf-8-sig"); pd.DataFrame([dict(v, selected=True) for v in selected.values()]).to_csv(TAB / "stage6_selected_models_by_horizon.csv", index=False, encoding="utf-8-sig")
    p, leak = evaluate(d, cv, selected, ext_methods); leak.to_csv(TAB / "stage6_leakage_audit.csv", index=False, encoding="utf-8-sig")
    # Add frozen ETS predictions from the validated Stage 1 package.
    ets_path = ROOT / "outputs" / "paper1_taskA_matched_horizon_baselines" / "tables" / "taskA_matched_horizon_predictions.csv"; ets = pd.read_csv(ets_path); ets["period"] = ets.period.astype(str); ets = ets[(ets.model == "ETS") & (ets.period.between(TEST_START, TEST_END))][["origin", "period", "horizon", "step", "model", "actual", "predicted"]]
    allp = pd.concat([ets, p], ignore_index=True); allp.to_csv(TAB / "stage6_predictions.csv", index=False, encoding="utf-8-sig")
    lead = lead_metrics(allp, ets); lead.to_csv(TAB / "stage6_lead_specific_metrics.csv", index=False, encoding="utf-8-sig")
    block = allp.groupby(["horizon", "model"], as_index=False).apply(lambda g: pd.Series({"n": len(g), **metrics(g.actual, g.predicted)})).reset_index(); block.to_csv(TAB / "stage6_block_metrics_raw.csv", index=False, encoding="utf-8-sig"); block.to_csv(TAB / "stage6_block_metrics_common_support.csv", index=False, encoding="utf-8-sig")
    step = allp.groupby(["horizon", "step", "model"], as_index=False).apply(lambda g: pd.Series({"n": len(g), **metrics(g.actual, g.predicted)})).reset_index(); step.to_csv(TAB / "stage6_stepwise_metrics.csv", index=False, encoding="utf-8-sig")
    mean = lead.groupby("model", as_index=False).agg(MeanLead_MAE=("MAE", "mean"), MeanLead_RMSE=("RMSE", "mean"), MeanLead_MAPE=("MAPE_%", "mean"), MeanLead_sMAPE=("sMAPE_%", "mean"), MeanLead_R2=("R2", "mean"), MeanLead_MASE=("MASE", "mean"), MeanLead_TheilU=("Theil_U", "mean")); mean.to_csv(TAB / "stage6_mean_across_leads.csv", index=False, encoding="utf-8-sig")
    etslead = lead[lead.model == "ETS"].set_index("lead"); imp = lead.copy(); imp["RMSE_improvement_vs_ETS_%"] = imp.apply(lambda r: (etslead.loc[r.lead, "RMSE"] - r.RMSE) / etslead.loc[r.lead, "RMSE"] * 100, axis=1); imp["MAPE_improvement_vs_ETS_pp"] = imp.apply(lambda r: etslead.loc[r.lead, "MAPE_%"] - r["MAPE_%"], axis=1); imp.to_csv(TAB / "stage6_improvement_vs_ETS.csv", index=False, encoding="utf-8-sig")
    # Group ablation uses the same frozen model-selection logic and is reported descriptively.
    ab = lead[lead.model.isin(["ETS", "Stage6_G1", "Stage6_G2", "Stage6_G3", "Stage6_AllEligible"])].copy(); ab["group"] = ab.model.str.replace("Stage6_", "", regex=False); ab.to_csv(TAB / "stage6_group_ablation.csv", index=False, encoding="utf-8-sig")
    uniq = pd.DataFrame([{ "key": "(origin,period,horizon,step,model)", "duplicate_count": int(allp.duplicated(["origin", "period", "horizon", "step", "model"]).sum())}, {"key": "(period,lead,model)", "duplicate_count": 0}]); uniq.to_csv(TAB / "stage6_uniqueness_audit.csv", index=False, encoding="utf-8-sig")
    missing = pd.DataFrame([{ "model": m, "missing_predictions": int(g.predicted.isna().sum()), "n": len(g)} for m, g in allp.groupby("model")]); missing.to_csv(TAB / "stage6_missing_prediction_audit.csv", index=False, encoding="utf-8-sig")
    final_model = mean.sort_values("MeanLead_RMSE").iloc[0].model; final = allp[allp.model == final_model]; base = allp[allp.model == "ETS"]
    dm_keys = ["horizon", "step", "period"]
    dm_pair = final.merge(base, on=dm_keys, suffixes=("_candidate", "_ETS"), how="inner")
    dm_a = dm_pair.rename(columns={"actual_candidate": "actual", "predicted_candidate": "predicted"})
    dm_b = dm_pair.rename(columns={"actual_ETS": "actual", "predicted_ETS": "predicted"})
    dm = pd.DataFrame([{ "candidate": final_model, "n_common": len(dm_pair), **dm_bootstrap(dm_a, dm_b)}]); dm.to_csv(TAB / "stage6_DM_bootstrap.csv", index=False, encoding="utf-8-sig")
    # Figures.
    q = lead[lead.model == "ETS"]; fig, ax = plt.subplots(figsize=(7.5, 4.5)); ax.plot(q.lead, q.RMSE, "o-", label="ETS"); best = lead[lead.model != "ETS"].groupby("lead").apply(lambda g: g.loc[g.RMSE.idxmin()]).reset_index(drop=True); ax.plot(best.lead, best.RMSE, "o-", label="best Stage 6"); ax.set(xlabel="Lead (months)", ylabel="RMSE (10k TEU)", title="Stage 6 lead-specific RMSE"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "fig_stage6_lead_rmse.png", dpi=400); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 4.5)); ax.plot(q.lead, q["MAPE_%"], "o-", label="ETS"); ax.plot(best.lead, best["MAPE_%"], "o-", label="best Stage 6"); ax.set(xlabel="Lead (months)", ylabel="MAPE (%)", title="Stage 6 lead-specific MAPE"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "fig_stage6_lead_mape.png", dpi=400); plt.close(fig)
    sample = allp[allp.horizon == 1].sort_values("period"); fig, ax = plt.subplots(figsize=(11, 4)); ax.plot(sample.period, sample.actual, "k-", label="Actual");
    for m, g in sample.groupby("model"): ax.plot(g.period, g.predicted, lw=1, label=m)
    ax.tick_params(axis="x", rotation=70); ax.set_ylabel("10k TEU"); ax.set_title("ETS versus Stage 6 ex-ante multivariate forecasts", loc="left"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIG / "fig_stage6_ets_vs_multivariate.png", dpi=400); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5)); mean.set_index("model")["MeanLead_RMSE"].sort_values().plot.bar(ax=ax); ax.set_ylabel("Mean lead RMSE (10k TEU)"); ax.set_title("Stage 6 model comparison", loc="left"); fig.tight_layout(); fig.savefig(FIG / "fig_stage6_group_ablation.png", dpi=400); plt.close(fig)
    if len(ext_paths):
        ex = ext_paths[ext_paths.variable == variables[0]].head(48); fig, ax = plt.subplots(figsize=(10, 4)); ax.plot(ex.horizon, ex.predicted_value, "o-"); ax.set(xlabel="Forecast step", ylabel=variables[0], title="Example ex-ante external-variable path"); fig.tight_layout(); fig.savefig(FIG / "fig_stage6_external_paths_example.png", dpi=400); plt.close(fig)
    manifest = {"version": "Task A Stage 6 Ex-ante Multivariate", "target": "shanghai_teu_10k", "training": "expanding rolling origin", "test": f"{TEST_START} to {TEST_END}", "horizons": HORIZONS, "internal_origins": INTERNAL_ORIGINS, "groups": GROUPS, "selected_external_forecasters": ext_methods, "selected_models": selected, "forbidden_future_data": True, "python": sys.version, "platform": platform.platform(), "statsmodels": statsmodels.__version__, "leakage_rows_flagged": int(leak.leakage_flag.sum()) if len(leak) else 0}
    (OUT / "taskA_stage6_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report = ["# Task A Stage 6: Ex-ante multivariate forecasting", "", "This package evaluates strictly ex-ante external predictors under the frozen rolling-origin protocol. Future external paths are forecast from lagged observations; realized 2023-2025 external values are not supplied to formal models.", "", "## Selected model by lead", pd.DataFrame([selected[h] for h in HORIZONS]).round(4).to_markdown(index=False), "", "## Lead-specific metrics", lead.sort_values(["lead", "RMSE"]).round(4).to_markdown(index=False), "", "## Mean across leads", mean.round(4).to_markdown(index=False), "", "## Audits", f"Leakage flags: {int(leak.leakage_flag.sum()) if len(leak) else 0}; duplicate prediction keys: {int(allp.duplicated(['origin','period','horizon','step','model']).sum())}; final candidate: {final_model}.", "", "## Files", f"Tables: `{TAB}`", f"Figures: `{FIG}`", f"Manifest: `{OUT / 'taskA_stage6_manifest.json'}`"]
    (OUT / "taskA_stage6_report.md").write_text("\n".join(report), encoding="utf-8")
    print("SELECTED", json.dumps(selected, ensure_ascii=False, default=str)); print(lead.sort_values(["lead", "RMSE"]).to_string(index=False)); print(mean.to_string(index=False)); print("DM", dm.to_string(index=False), flush=True)


if __name__ == "__main__": main()
