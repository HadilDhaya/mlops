
# ===== 1. Import the libraries =====
import mlflow
from mlflow import MlflowClient

# ===== 2. Connect to the same database as Lab 3 =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ===== 3. Find the best run (same logic as in Lab 3) =====
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

best = runs.iloc[0]
run_id = best["run_id"]

print("Best run :", best["tags.mlflow.runName"], "| id :", run_id[:8], "...")

# ===== 4. Register the model of this run in the Registry =====
result = mlflow.register_model(
    f"runs:/{run_id}/model",
    "churn-model"
)

print("Model registered : churn-model | version", result.version)

# ===== 5. Put the alias "champion" on this version =====
client = MlflowClient()

client.set_registered_model_alias(
    "churn-model",
    "champion",
    result.version
)

print("Alias 'champion' -> version", result.version)
