from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

def load_data(save_path="data/raw_data.csv"):
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    os.makedirs("data", exist_ok=True)
    df.to_csv(save_path, index=False)

    return df