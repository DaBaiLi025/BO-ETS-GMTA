from __future__ import annotations

import json
import os
import platform
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "clean_model_dataset_v5_shanghai_augmented.xlsx"
STAGE2 = ROOT / "outputs" / "paper1_taskA_stage2_traditional_models" / "tables" / "taskA_stage2_predictions.csv"
OUT = ROOT / "outputs" / "paper1_taskA_stage3_raw_tcn_mimo_pytorch_cuda"
TAB, FIG = OUT / "tables", OUT / "figures"
for p in (OUT, TAB, FIG): p.mkdir(parents=True, exist_ok=True)
TEST_START, TEST_END = "2023-01", "2025-12"
HORIZONS = [1, 3, 6, 12]
SEEDS = [42, 52, 62]
INTERNAL_ORIGINS = ["2018-12", "2019-12", "2020-12", "2021-12"]
CONFIGS = [
    {"lookback": 12, "filters": 8, "kernel_size": 2, "dilations": [1, 2, 4], "dropout": 0.0, "residual_blocks": 1},
    {"lookback": 24, "filters": 8, "kernel_size": 2, "dilations": [1, 2, 4], "dropout": 0.1, "residual_blocks": 1},
    {"lookback": 24, "filters": 16, "kernel_size": 3, "dilations": [1, 2, 4], "dropout": 0.1, "residual_blocks": 2},
    {"lookback": 36, "filters": 16, "kernel_size": 2, "dilations": [1, 2, 4, 8], "dropout": 0.1, "residual_blocks": 2},
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def load():
    d = pd.read_excel(DATA, sheet_name="Clean_Main_2012_2025")
    d.period = d.period.astype(str); d = d.sort_values("period").drop_duplicates("period")
    d["y"] = pd.to_numeric(d.shanghai_teu_10k, errors="coerce")
    return d[d.y.notna()].reset_index(drop=True)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float); den = np.abs(y) + np.abs(p)
    return {"MAE": float(np.mean(np.abs(y-p))), "RMSE": float(np.sqrt(np.mean((y-p)**2))), "MAPE_%": float(np.mean(np.abs((y-p)/y))*100), "sMAPE_%": float(np.mean(np.where(den == 0, 0, 2*np.abs(y-p)/den))*100), "R2": float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2))}


def make_samples(values, lookback, horizon, origin_idx):
    xs, ys, audit = [], [], []
    for end in range(lookback - 1, origin_idx - horizon + 1):
        xs.append(values[end-lookback+1:end+1]); ys.append(values[end+1:end+1+horizon])
        audit.append({"forecast_origin": int(origin_idx), "horizon": horizon, "lookback": lookback, "max_training_input_end": int(end), "max_training_output_end": int(end+horizon), "forecast_start": int(origin_idx+1), "forecast_end": int(origin_idx+horizon), "leakage_flag": bool(end+horizon > origin_idx)})
    return np.asarray(xs, float), np.asarray(ys, float), audit


class CausalConv1d(nn.Module):
    def __init__(self, cin, cout, kernel, dilation):
        super().__init__(); self.left = (kernel - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, kernel, dilation=dilation)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.left, 0)))


class TCN(nn.Module):
    def __init__(self, cfg, horizon):
        super().__init__(); blocks = []; cin = 1
        for _ in range(cfg["residual_blocks"]):
            layers = []
            for dilation in cfg["dilations"]:
                layers += [CausalConv1d(cin, cfg["filters"], cfg["kernel_size"], dilation), nn.ReLU()]
                if cfg["dropout"]: layers.append(nn.Dropout(cfg["dropout"]))
                cin = cfg["filters"]
            blocks.append(nn.ModuleList(layers))
        self.blocks = nn.ModuleList(blocks); self.projections = nn.ModuleList()
        for i in range(cfg["residual_blocks"]): self.projections.append(nn.Conv1d(1 if i == 0 else cfg["filters"], cfg["filters"], 1))
        self.head = nn.Linear(cfg["lookback"] * cfg["filters"], horizon)

    def forward(self, x):
        for block, proj in zip(self.blocks, self.projections):
            residual = x
            for layer in block: x = layer(x)
            if residual.shape[1] != x.shape[1]: residual = proj(residual)
            x = torch.relu(x + residual)
        return self.head(x.flatten(1))


def fit_predict(values, origin_idx, horizon, cfg, seed, epochs=140):
    seed_everything(seed); scaler = StandardScaler().fit(np.asarray(values[:origin_idx+1]).reshape(-1, 1))
    scaled = scaler.transform(np.asarray(values[:origin_idx+1]).reshape(-1, 1)).ravel()
    X, Y, audit = make_samples(scaled, cfg["lookback"], horizon, origin_idx)
    if len(X) < 12: raise ValueError("fewer than 12 causal training samples")
    split = max(1, int(len(X) * 0.8)); xtr, xv = X[:split], X[split:]; ytr, yv = Y[:split], Y[split:]
    xt = torch.tensor(xtr[:, None, :], dtype=torch.float32, device=DEVICE); yt = torch.tensor(ytr, dtype=torch.float32, device=DEVICE)
    xval = torch.tensor(xv[:, None, :], dtype=torch.float32, device=DEVICE) if len(xv) else None
    yval = torch.tensor(yv, dtype=torch.float32, device=DEVICE) if len(yv) else None
    model = TCN(cfg, horizon).to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=0.001); loss_fn = nn.MSELoss()
    best_state, best_loss, patience = None, float("inf"), 0
    for _ in range(epochs):
        model.train(); opt.zero_grad(set_to_none=True); loss = loss_fn(model(xt), yt); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): val_loss = float(loss_fn(model(xval), yval).item()) if xval is not None else float(loss.item())
        if val_loss < best_loss - 1e-5:
            best_loss, patience = val_loss, 0; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else: patience += 1
        if patience >= 15: break
    if best_state is not None: model.load_state_dict(best_state)
    last = torch.tensor(scaled[-cfg["lookback"]:][None, None, :], dtype=torch.float32, device=DEVICE)
    with torch.no_grad(): pred_scaled = model.eval()(last).detach().cpu().numpy().ravel()
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    return np.maximum(pred, 0.0), audit, sum(p.numel() for p in model.parameters()), model.count_parameters() if hasattr(model, "count_parameters") else sum(p.numel() for p in model.parameters())


def tune(d, horizon):
    rows = []
    for ci, cfg in enumerate(CONFIGS):
        fold = []
        for origin in INTERNAL_ORIGINS:
            oi = int(d.index[d.period == origin][0]); values = d.y.iloc[:oi+1].to_numpy(float); future = d.y.iloc[oi+1:oi+1+horizon].to_numpy(float)
            if len(future) < horizon: continue
            try:
                pred, _, _, params = fit_predict(values, oi, horizon, cfg, 42, epochs=100); fold.append({"origin": origin, **metrics(future, pred)})
            except Exception as e:
                rows.append({"horizon": horizon, "config_id": ci, "config": json.dumps(cfg), "status": "failed", "error": str(e)}); fold=[]; break
        if fold: rows.append({"horizon": horizon, "config_id": ci, "config": json.dumps(cfg), "status": "ok", "mean_RMSE": np.mean([x["RMSE"] for x in fold]), "sd_RMSE": np.std([x["RMSE"] for x in fold], ddof=1), "mean_MAPE_%": np.mean([x["MAPE_%"] for x in fold]), "n_folds": len(fold), "parameter_count": params})
    cv = pd.DataFrame(rows); ok = cv[cv.status == "ok"].sort_values(["mean_RMSE", "mean_MAPE_%"])
    if ok.empty: raise RuntimeError(f"No valid TCN configuration for H={horizon}")
    return ok.iloc[0].to_dict(), cv


def stage2_audit():
    d = pd.read_csv(STAGE2); d.period = d.period.astype(str); expected = []
    for h in HORIZONS:
        for origin in sorted(d.origin.unique()):
            for step in range(1, h+1):
                p = str(pd.Period(origin, freq="M") + step)
                if TEST_START <= p <= TEST_END: expected.append((origin, p, h, step))
    models = sorted(d.model.unique()); rows = []
    for origin, period, h, step in sorted(set(expected)):
        g = d[(d.origin == origin) & (d.period == period) & (d.horizon == h) & (d.step == step)]
        for model in models:
            if len(g[g.model == model]) == 0: rows.append({"origin": origin, "period": period, "horizon": h, "step": step, "model": model, "missing": True, "reason": "source file has no prediction record"})
    pd.DataFrame(rows).to_csv(TAB / "taskA_stage2_missing_prediction_audit.csv", index=False, encoding="utf-8-sig")
    key = ["origin", "period", "horizon", "step"]; common = d.groupby(key).model.nunique(); common_keys = common[common == len(models)].index; common_d = d.set_index(key).loc[common_keys].reset_index()
    for label, frame in [("raw", d), ("common_support", common_d)]:
        out = []
        for (h, m), g in frame.groupby(["horizon", "model"]): out.append({"block_horizon": h, "model": m, "support": "all" if label == "raw" else "common", "n": len(g), **metrics(g.actual, g.predicted)})
        pd.DataFrame(out).to_csv(TAB / f"taskA_stage2_block_metrics_{label}.csv", index=False, encoding="utf-8-sig")
    return len(rows), len(common_d)


def main():
    d = load(); missing, common_n = stage2_audit(); periods = d.period.tolist(); first = periods.index(TEST_START); last = periods.index(TEST_END)
    selected, cv_all = [], []
    for h in HORIZONS:
        best, cv = tune(d, h); selected.append(best); cv_all.append(cv)
    sel = pd.DataFrame(selected); pd.concat(cv_all, ignore_index=True).to_csv(TAB / "taskA_stage3_internal_cv_records.csv", index=False, encoding="utf-8-sig"); sel.to_csv(TAB / "taskA_stage3_selected_hyperparameters.csv", index=False, encoding="utf-8-sig")
    pmap = {int(r.horizon): json.loads(r.config) for _, r in sel.iterrows()}; seeds, leak = [], []
    for h in HORIZONS:
        cfg = pmap[h]
        for oi in range(max(cfg["lookback"] - 1, first - h), last):
            origin = periods[oi]; future = d.iloc[oi+1:oi+1+h]
            if len(future) < h: continue
            for seed in SEEDS:
                try: pred, audits, _, nparams = fit_predict(d.y.iloc[:oi+1].to_numpy(float), oi, h, cfg, seed); status, err = "ok", ""
                except Exception as e: pred, audits, nparams, status, err = np.repeat(d.y.iloc[oi], h), [], 0, "failed", str(e)
                for a in audits: leak.append({"forecast_origin": origin, "horizon": h, "lookback": cfg["lookback"], "max_training_input_end": a["max_training_input_end"], "max_training_output_end": a["max_training_output_end"], "leakage_flag": a["leakage_flag"]})
                for j, (_, row) in enumerate(future.iterrows(), 1):
                    if TEST_START <= row.period <= TEST_END: seeds.append({"origin": origin, "period": row.period, "horizon": h, "step": j, "model": "RawTCN_MIMO_PyTorch_CUDA", "seed": seed, "actual": float(row.y), "predicted": float(pred[j-1]), "status": status, "error": err, "parameter_count": nparams})
    sd = pd.DataFrame(seeds); sd.to_csv(TAB / "taskA_stage3_seed_predictions.csv", index=False, encoding="utf-8-sig")
    final = sd[sd.status == "ok"].groupby(["origin", "period", "horizon", "step", "model", "actual"], as_index=False).predicted.mean().rename(columns={"predicted": "predicted_final"}); final.to_csv(TAB / "taskA_stage3_predictions.csv", index=False, encoding="utf-8-sig")
    seed_metrics = [{"horizon": h, "seed": s, "n": len(g), **metrics(g.actual, g.predicted)} for (h, s), g in sd[sd.status == "ok"].groupby(["horizon", "seed"])]
    pd.DataFrame(seed_metrics).to_csv(TAB / "taskA_stage3_seed_metrics.csv", index=False, encoding="utf-8-sig")
    lead_rows, block_rows, step_rows = [], [], []
    for h in HORIZONS:
        for model, g in final[final.horizon == h].groupby("model"):
            lead_rows.append({"lead": h, "model": model, "n": len(g[g.step == h]), **metrics(g[g.step == h].actual, g[g.step == h].predicted_final)})
            block_rows.append({"block_horizon": h, "model": model, "n": len(g), **metrics(g.actual, g.predicted_final)})
            for step, z in g.groupby("step"): step_rows.append({"block_horizon": h, "step": step, "model": model, "n": len(z), **metrics(z.actual, z.predicted_final)})
    lead = pd.DataFrame(lead_rows); block = pd.DataFrame(block_rows); step = pd.DataFrame(step_rows)
    lead.to_csv(TAB / "taskA_stage3_lead_specific_metrics.csv", index=False, encoding="utf-8-sig"); block.to_csv(TAB / "taskA_stage3_block_metrics_raw.csv", index=False, encoding="utf-8-sig"); step.to_csv(TAB / "taskA_stage3_stepwise_metrics.csv", index=False, encoding="utf-8-sig")
    mean = lead.groupby("model").agg(MeanLead_MAE=("MAE", "mean"), MeanLead_RMSE=("RMSE", "mean"), MeanLead_MAPE=("MAPE_%", "mean"), MeanLead_sMAPE=("sMAPE_%", "mean"), MeanLead_R2=("R2", "mean")).reset_index(); mean.to_csv(TAB / "taskA_stage3_mean_across_leads.csv", index=False, encoding="utf-8-sig")
    ets = pd.read_csv(ROOT / "outputs/paper1_taskA_stage2_traditional_models/tables/taskA_stage2_lead_specific_metrics.csv"); ets = ets[ets.model == "ETS"].set_index("lead"); imp = lead.copy(); imp["RMSE_improvement_vs_ETS_%"] = imp.apply(lambda r: (ets.loc[r.lead, "RMSE"] - r.RMSE) / ets.loc[r.lead, "RMSE"] * 100, axis=1); imp["MAPE_improvement_vs_ETS_pp"] = imp.apply(lambda r: ets.loc[r.lead, "MAPE_%"] - r["MAPE_%"], axis=1); imp.to_csv(TAB / "taskA_stage3_improvement_vs_ETS.csv", index=False, encoding="utf-8-sig")
    key = final.groupby(["period", "horizon", "step", "model"]).size().reset_index(name="n_rows"); key.to_csv(TAB / "taskA_stage3_uniqueness_audit.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(leak).to_csv(TAB / "taskA_stage3_leakage_audit.csv", index=False, encoding="utf-8-sig")
    for col, name, color in [("RMSE", "RMSE", "#4E79A7"), ("MAPE_%", "MAPE", "#F28E2B")]:
        fig, ax = plt.subplots(figsize=(7.5, 4.3)); q = lead.sort_values("lead"); ax.plot(q.lead, q[col], marker="o", lw=1.6, color=color, label="Raw TCN-MIMO PyTorch CUDA"); e = [float(ets.loc[h, col]) for h in HORIZONS]; ax.plot(HORIZONS, e, marker="o", lw=1.6, color="#59A14F", label="ETS"); ax.set_xlabel("Forecast lead (months)"); ax.set_ylabel(col); ax.set_title(f"Task A Stage 3 lead-specific {name}", loc="left", fontweight="bold"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / f"fig_stage3_lead_{name.lower()}.png", dpi=400); fig.savefig(FIG / f"fig_stage3_lead_{name.lower()}.svg"); plt.close(fig)
    manifest = {"version": "Task A Stage 3 Raw TCN-MIMO PyTorch CUDA", "target": "shanghai_teu_10k", "test": f"{TEST_START} to {TEST_END}", "horizons": HORIZONS, "seeds": SEEDS, "internal_origins": INTERNAL_ORIGINS, "configs": CONFIGS, "selected": sel.to_dict(orient="records"), "training": "strict causal expanding origin", "scaler": "StandardScaler fit only on values through each forecast origin", "aggregation": "arithmetic mean across seeds; no best-seed selection", "external_variables": False, "decomposition": False, "device": str(DEVICE), "pytorch": torch.__version__, "cuda": torch.version.cuda, "stage2_missing_records": int(missing), "stage2_common_support_records": int(common_n), "python": platform.python_version()}
    (OUT / "taskA_stage3_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report = ["# Task A Stage 3: Raw TCN-MIMO (PyTorch CUDA)", "", f"Device: `{DEVICE}`; PyTorch: `{torch.__version__}`; CUDA: `{torch.version.cuda}`.", "", "Raw TCN-MIMO uses only historical Shanghai Port TEU values. Four independently trained horizon tasks (H=1,3,6,12) use a small causal TCN and a Dense(H) multi-output head. Architecture selection uses only internal origins no later than 2021-12. Final predictions average seeds 42, 52 and 62.", "", "## Selected configurations", "", sel[["horizon", "config_id", "config", "mean_RMSE", "mean_MAPE_%", "parameter_count"]].round(4).to_markdown(index=False), "", "## Lead-specific results", "", lead.round(4).to_markdown(index=False), "", "## Mean across leads", "", mean.round(4).to_markdown(index=False), "", "## Audits", "", f"Stage-2 missing support records audited: {missing}.", f"Stage-2 common-support rows retained: {common_n}.", f"TCN duplicate key rows: {int((key.n_rows > 1).sum())}.", f"TCN leakage flags: {int(pd.DataFrame(leak).leakage_flag.sum()) if leak else 0}."]
    (OUT / "taskA_stage3_raw_tcn_mimo_report.md").write_text("\n".join(report), encoding="utf-8"); print(lead.to_string(index=False)); print("MEAN\n", mean.to_string(index=False), flush=True)


if __name__ == "__main__": main()
