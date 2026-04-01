import numpy as np
import optuna
from utils.metrics import rmse

def optimize_weights(preds_list, y_true):

    def objective(trial):
        weights = [
            trial.suggest_float(f"w{i}", 0, 1)
            for i in range(len(preds_list))
        ]

        weights = np.array(weights)
        weights /= weights.sum()

        final_pred = sum(w * p for w, p in zip(weights, preds_list))

        return rmse(y_true, final_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    return study