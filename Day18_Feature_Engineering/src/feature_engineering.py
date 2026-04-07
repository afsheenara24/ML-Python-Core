import pandas as pd
import os

def create_features(df, save_path="data/processed_data.csv"):
    df = df.copy()

    df["rooms_per_household"] = df["AveRooms"] / df["HouseAge"]
    df["bedrooms_ratio"] = df["AveBedrms"] / df["AveRooms"]
    df["income_rooms_interaction"] = df["MedInc"] * df["AveRooms"]

    # Save processed data
    os.makedirs("data", exist_ok=True)
    df.to_csv(save_path, index=False)

    return df