#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


# In[2]:


def find_best_weights(models, X_val, y_val):
    """
    Find optimal weights using grid search
    """
    best_score = float("inf")
    best_weights = None

    model_list = list(models.values())

    for w1 in np.linspace(0, 1, 5):
        for w2 in np.linspace(0, 1, 5):
            for w3 in np.linspace(0, 1, 5):

                weights = np.array([w1, w2, w3])

                if weights.sum() == 0:
                    continue

                weights = weights / weights.sum()

                preds = np.column_stack([
                    model.predict(X_val) for model in model_list
                ])

                final_preds = np.dot(preds, weights)

                score = mean_squared_error(y_val, final_preds)

                if score < best_score:
                    best_score = score
                    best_weights = weights

    return best_weights, best_score


# In[3]:


def tune_random_forest(X, y):
    """
    Hyperparameter tuning for RandomForest
    """
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5, None]
    }

    grid = GridSearchCV(
        RandomForestRegressor(),
        param_grid,
        cv=3,
        scoring="neg_mean_squared_error"
    )

    grid.fit(X, y)

    return grid.best_estimator_


# In[ ]:




