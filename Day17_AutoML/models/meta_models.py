import optuna
import numpy as np
from sklearn.linear_model import Ridge
from utils.metrics import rmse
from utils.cv import get_cv

def optimize_meta(X_meta, y):

    def objective(trial):
        params = meta_space(trial)
        model = Ridge(**params)

        scores = []
        cv = get_cv()

        for train_idx, val_idx in cv.split(X_meta):
            X_tr, X_val = X_meta[train_idx], X_meta[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)

            scores.append(rmse(y_val, preds))

        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    return study