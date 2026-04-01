from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

def get_rf(params):
    return RandomForestRegressor(**params)

def get_gb(params):
    return GradientBoostingRegressor(**params)