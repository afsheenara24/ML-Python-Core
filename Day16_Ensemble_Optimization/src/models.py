#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# In[2]:


def get_base_models():
    """
    Returns dictionary of base models
    """
    return {
        "lr": LinearRegression(),
        "rf": RandomForestRegressor(n_estimators=100, random_state=42),
        "xgb": XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    }


# In[3]:


def train_models(models, X, y):
    """
    Train all models and return trained versions
    """
    trained_models = {}

    for name, model in models.items():
        model.fit(X, y)
        trained_models[name] = model

    return trained_models


# In[ ]:




