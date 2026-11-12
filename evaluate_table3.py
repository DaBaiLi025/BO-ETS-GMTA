from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
START, END = "2023-01", "2025-12"
LEADS = [1, 3, 6, 12]


def load_predictions(path: Path, model: str, prediction_col: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["period"] = frame["period"].astype(str)
    if "step" in frame and "horizon" in frame:
        frame = frame[frame["step"] == frame["horizon"]]
    frame = frame[frame["period"].between(START, END)].copy()
    frame = frame.rename(columns={prediction_col: "prediction"})
    frame["model"] = model
    if "lead" not in frame:
        frame["lead"] = frame["horizon"]
    return frame[["origin", "period", "lead", "actual", "prediction", "model"]]


def collect_predictions() -> pd.DataFrame:
    out = ROOT / "outputs"
    frames = []

    stage2 = pd.read_csv(
        out / "paper1_taskA_stage2_traditional_models/tables/taskA_stage2_predictions.csv"
    )
    stage2["period"] = stage2["period"].astype(str)
    stage2 = stage2[
        (stage2["period"].between(START, END))
        & (stage2["horizon"] == stage2["step"])
    ].copy()
    stage2_names = {
        "SeasonalNaive": "Seasonal Naive",
        "ETS": "ETS",
        "SARIMA": "SARIMA",
        "Ridge": "Ridge",
        "ElasticNet": "ElasticNet",
        "SVR": "SVR",
    }
    for source_name, model in stage2_names.items():
        q = stage2[stage2["model"] == source_name].copy()
        q["lead"] = q["horizon"]
        q["model"] = model
        q = q.rename(columns={"predicted": "prediction"})
        frames.append(q[["origin", "period", "lead", "actual", "prediction", "model"]])

    frames.append(
        load_predictions(
            out / "paper1_bayesian_optimization/predictions/bo_ets_s_predictions.csv",
            "BO-ETS-S",
                "prediction",
        )
    )

    gmta = pd.read_csv(
        out / "paper1_bo_ets_residual_v2/predictions/gated_multiscale_predictions.csv"
    )
    gmta["period"] = gmta["period"].astype(str)
    gmta = gmta[gmta["period"].between(START, END)].copy()
    gmta["lead"] = gmta["horizon"]
    q = gmta[["origin", "period", "lead", "actual", "final_prediction"]].copy()
    q = q.rename(columns={"final_prediction": "prediction"})
    q["model"] = "BO-ETS-GMTA"
    frames.append(q)

    for filename, model in [
        (
            "paper1_taskA_stage3_raw_tcn_mimo_pytorch_cuda/tables/taskA_stage3_predictions.csv",
            "Raw TCN",
        ),
        (
            "paper1_taskA_stage5_standard_vmd_tcn_mimo_pytorch_cuda/tables/taskA_stage5_predictions.csv",
            "VMD-TCN",
        ),
    ]:
        col = "predicted_final"
        frames.append(load_predictions(out / filename, model, col))

    stage6 = pd.read_csv(out / "paper1_taskA_stage6_exante_multivariate/tables/stage6_predictions.csv")
    stage6["period"] = stage6["period"].astype(str)
    stage6 = stage6[
        (stage6["model"] == "Stage6Selected")
        & (stage6["period"].between(START, END))
        & (stage6["horizon"] == stage6["step"])
    ].copy()
    stage6["lead"] = stage6["horizon"]
    stage6["model"] = "Ex-ante Multivariate"
    stage6 = stage6.rename(columns={"predicted": "prediction"})
    frames.append(stage6[["origin", "period", "lead", "actual", "prediction", "model"]])

    predictions = pd.concat(frames, ignore_index=True)
    key = ["origin", "period", "lead", "model"]
    duplicates = predictions.duplicated(key)
    if duplicates.any():
        raise ValueError(f"Duplicate prediction keys found: {int(duplicates.sum())}")
    counts = predictions.groupby(["model", "lead"]).size()
    if not counts.eq(36).all() or set(counts.index.get_level_values("lead")) != set(LEADS):
        raise ValueError("Each Table 3 model must have exactly 36 predictions at every lead.")
    return predictions


def metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, lead), group in predictions.groupby(["model", "lead"]):
        actual = group["actual"].to_numpy(float)
        predicted = group["prediction"].to_numpy(float)
        error = actual - predicted
        rows.append(
            {
                "model": model,
                "lead": int(lead),
                "RMSE": float(np.sqrt(np.mean(error**2))),
                "MAE": float(np.mean(np.abs(error))),
                "MAPE (%)": float(np.mean(np.abs(error / actual)) * 100),
                "sMAPE (%)": float(
                    np.mean(2 * np.abs(error) / (np.abs(actual) + np.abs(predicted))) * 100
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    predictions = collect_predictions()
    metrics = metric_rows(predictions)
    order = [
        "BO-ETS-GMTA",
        "BO-ETS-S",
        "ETS",
        "Ridge",
        "Ex-ante Multivariate",
        "SARIMA",
        "ElasticNet",
        "Raw TCN",
        "VMD-TCN",
        "Seasonal Naive",
        "SVR",
    ]
    wide = metrics.pivot(index="model", columns="lead")
    expected_models = set(order)
    if set(wide.index) != expected_models:
        raise ValueError(f"Expected 11 Table 3 models, found: {sorted(wide.index)}")
    table = pd.DataFrame(index=order)
    for lead in LEADS:
        table[f"RMSE / H{lead}"] = wide["RMSE"][lead]
    for measure in ["RMSE", "MAE", "MAPE (%)", "sMAPE (%)"]:
        table[f"MeanLead / {measure}"] = wide[measure].mean(axis=1)
    table.index.name = "Model"
    table = table.reset_index()

    output = ROOT / "outputs" / "table3-code-release"
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "table3_predictions.csv", index=False)
    metrics.to_csv(output / "table3_lead_metrics.csv", index=False)
    table.to_csv(output / "table3_metrics.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
