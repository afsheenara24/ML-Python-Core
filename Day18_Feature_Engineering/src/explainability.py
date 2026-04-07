import shap
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import numpy as np

def shap_importance(model, X_train):
    print("Calculating SHAP (sampling 1000 rows)...")

    # Sample data (VERY IMPORTANT)
    sample_idx = np.random.choice(X_train.shape[0], 1000, replace=False)
    X_sample = X_train[sample_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig("outputs/shap_importance.png")

    print("SHAP completed and saved.")

def permutation_importance_calc(model, X_test, y_test):
    result = permutation_importance(model, X_test, y_test, n_repeats=10)

    return result.importances_mean