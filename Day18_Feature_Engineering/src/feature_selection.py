import numpy as np

def select_features(importances, threshold=0.01):
    selected_idx = np.where(importances > threshold)[0]
    return selected_idx