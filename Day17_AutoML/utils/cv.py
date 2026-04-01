from sklearn.model_selection import KFold

def get_cv():
    return KFold(n_splits=5, shuffle=True, random_state=42)