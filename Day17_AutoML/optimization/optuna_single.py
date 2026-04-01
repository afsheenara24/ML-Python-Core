import optuna
import numpy as np
from utils.cv import get_cv
from utils.metrics import rmse
from models.base_models import get_rf
from optimization.search_space import rf_space

def optimize_rf(X, y):

    def objective(trial):
        params = rf_space(trial)
        model = get_rf(params)
    
        scores = []
        cv = get_cv()
    
        for step, (train_idx, val_idx) in enumerate(cv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
    
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
    
            score = rmse(y_val, preds)
            scores.append(score)
    
            # Report progress
            trial.report(np.mean(scores), step=step)
    
            # Prune bad trials
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    return study