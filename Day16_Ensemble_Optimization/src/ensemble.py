#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def get_model_predictions(models, X):
    """
    Get predictions from all models
    """
    preds = []

    for model in models.values():
        preds.append(model.predict(X))

    return np.column_stack(preds)


# In[3]:


def weighted_average(models, X, weights):
    """
    Compute weighted average predictions
    """
    preds = get_model_predictions(models, X)

    weights = np.array(weights)
    weights = weights / weights.sum()

    final_preds = np.dot(preds, weights)

    return final_preds


# In[4]:


def blending_train(meta_model, base_models, X_val, y_val):
    """
    Train meta-model on validation predictions
    """
    base_preds = get_model_predictions(base_models, X_val)
    meta_model.fit(base_preds, y_val)

    return meta_model


# In[5]:


def blending_predict(meta_model, base_models, X):
    """
    Predict using blending
    """
    base_preds = get_model_predictions(base_models, X)
    return meta_model.predict(base_preds)


# In[ ]:




