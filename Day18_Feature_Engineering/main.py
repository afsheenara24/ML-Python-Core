from src.load_data import load_data
from src.preprocessing import preprocess
from src.feature_engineering import create_features
from src.model import train_model
from src.explainability import shap_importance, permutation_importance_calc
from src.feature_selection import select_features
from src.dimensionality_reduction import apply_pca

def main():
    print("Loading data...")
    df = load_data()   # saves raw_data.csv

    print("Creating features...")
    df = create_features(df)  # saves processed_data.csv

    print("Preprocessing...")
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Running SHAP...")
    shap_importance(model, X_train)
    print("SHAP done!")

    print("Computing permutation importance...")
    importances = permutation_importance_calc(model, X_test, y_test)

    print("Selecting features...")
    selected_idx = select_features(importances)

    X_train = X_train[:, selected_idx]
    X_test = X_test[:, selected_idx]

    print("Applying PCA...")
    X_train_pca, X_test_pca, pca = apply_pca(X_train, X_test)

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()