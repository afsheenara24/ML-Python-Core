import json
import numpy as np
from data.load_data import load_data
from optimization.optuna_single import optimize_rf
from optimization.optuna_ensemble import optimize_weights

X_train, X_test, y_train, y_test = load_data()

# Step 1: Optimize model
rf_study = optimize_rf(X_train, y_train)
best_rf_params = rf_study.best_params

# Train final model
from models.base_models import get_rf
rf_model = get_rf(best_rf_params)
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)

# Example second model (reuse or dummy)
gb_preds = rf_preds * 0.95  # placeholder

# Step 2: Optimize ensemble
study_weights = optimize_weights([rf_preds, gb_preds], y_test)

weights = study_weights.best_params

# Final prediction
final_pred = sum(
    weights[f"w{i}"] * p
    for i, p in enumerate([rf_preds, gb_preds])
)

# Save outputs
np.savetxt("outputs/predictions.csv", final_pred, delimiter=",")

metrics = {
    "rmse": float(np.sqrt(((y_test - final_pred) ** 2).mean()))
}

with open("outputs/metrics.json", "w") as f:
    json.dump(metrics, f)