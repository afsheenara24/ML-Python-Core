from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

def build_pipeline():
    pipeline = Pipeline([
        ("model", RandomForestRegressor(n_estimators=100))
    ])
    return pipeline