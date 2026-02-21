# ML-Python-Core

## Day 1 – Python for Machine Learning

### Topics covered:
- Python basics for ML (functions, list comprehension, lambda)
- NumPy fundamentals (ndarray, vectorization, broadcasting)
- Performance comparison: loops vs vectorized operations
- Pandas basics (DataFrame, data inspection, cleaning)
- Simple feature engineering


## Day 2 – Advanced NumPy & Linear Regression

### Topics Covered
- Advanced NumPy operations and vectorization
- Exploratory Data Analysis (EDA)
- Data visualization using Matplotlib
- Train-test split
- Linear Regression model training
- Model evaluation using MSE and R²
- Underfitting analysis

### Key Learnings
- Visualization helps in selecting appropriate models
- Linear regression can underfit when features are insufficient
- Evaluation metrics confirm model limitations

##  Day 3 – Polynomial Regression & Model Evaluation

### Topics Covered
- Polynomial feature transformation
- Linear vs Polynomial Regression
- Train-test split for model validation
- Model performance comparison
- Evaluation metrics: R², MAE, RMSE
- Bias–variance tradeoff
- Overfitting and underfitting analysis

### Key Learnings
- Polynomial regression captures non-linear relationships better than linear models
- Higher model complexity can lead to overfitting if not controlled
- Train-test split helps detect generalization issues
- R² alone is insufficient to evaluate model performance.
- Error-based metrics like MAE and RMSE provide real-world insight into prediction accuracy

## Day 4 – Multiple Linear Regression

### Topics Covered
- Introduction to Multiple Linear Regression (MLR)
- Mathematical formulation of MLR
- Assumptions of linear regression
- Feature scaling and normalization
- Multiple Linear Regression from scratch using NumPy
- Gradient Descent optimization
- Train–test split and data leakage prevention
- Multiple Linear Regression using scikit-learn
- Model evaluation using MSE and R²
- Residual analysis
- Multicollinearity detection using Variance Inflation Factor (VIF)
- Interpretation of regression coefficients

### Key Learnings
- Multiple features improve model expressiveness compared to simple linear regression
- Feature scaling is essential for stable and faster convergence
- Train–test split ensures reliable performance evaluation
- Residual analysis helps validate model assumptions
- Multicollinearity negatively impacts coefficient stability and interpretability
- Regression coefficients provide actionable insights into feature importance

## Day 5 - Polynomial Regression & Bias–Variance Tradeoff

### Topics Covered

- Polynomial Regression
- Feature expansion using Polynomial Features
- Model complexity and flexibility
- Bias–Variance Tradeoff
- Underfitting vs Overfitting analysis
- Effect of polynomial degree on model performance
- Visualization of model fits for different degrees

### Key Learnings

- Polynomial regression captures non-linear relationships while remaining linear in parameters
- Increasing model complexity reduces bias but increases variance
- Low-degree models tend to underfit, while high-degree models tend to overfit
- Visualization is crucial for understanding bias–variance behavior
- Optimal model performance lies in balancing bias and variance for better generalization

## Day 6 – Regularization Techniques

### Topics Covered
- Ridge Regression (L2 Regularization)
- Lasso Regression (L1 Regularization)
- Effect of alpha on model complexity
- Feature scaling importance

### Key Learnings
- Regularization helps reduce overfitting
- Ridge shrinks coefficients but keeps all features
- Lasso performs feature selection
- Elastic Net combines Ridge and Lasso benefits

## Day 7 – Cross-Validation & Hyperparameter Tuning

### Topics Covered

- Limitations of train–test split
- K-Fold Cross-Validation
- Bias–variance perspective of model evaluation
- Hyperparameter tuning concepts
- GridSearchCV
- RandomizedSearchCV
- Regularization tuning for Ridge, Lasso, and Elastic Net
- ML Pipelines 
- Data leakage prevention

### Key Learnings

- Single train–test split provides a high-variance estimate of model performance
- K-Fold Cross-Validation yields more reliable and stable evaluation results
- Proper hyperparameter tuning improves generalization performance
- Pipelines are essential for preventing data leakage during cross-validation
- RandomizedSearchCV is computationally efficient for large hyperparameter spaces
- Cross-validation is critical for production-ready machine learning models

## Day 8 – Feature Engineering & Feature Selection

### Topics Covered
- Feature scaling and power transformations
- Handling skewed numerical features
- Encoding categorical variables
- Feature selection using RFE (Wrapper Method)
- Feature selection using L1 regularization (Embedded Method)
- Pipelines and cross-validation to avoid data leakage

### Key Learnings
- Feature engineering has a greater impact than model choice
- Feature selection must be done after proper preprocessing
- Pipelines are essential for production-ready ML systems

## Day 9 – Decision Trees

### Topics Covered

- Intuition behind decision tree splitting
- Impurity measures: Gini, Entropy, and MSE
- Decision Trees for classification and regression
- Overfitting in trees and bias–variance behavior
- Tree regularization parameters (max_depth, min_samples_leaf, min_samples_split)
- Handling categorical features using pipelines
- Feature importance and its limitations
- Hyperparameter tuning using GridSearchCV
- Bias–variance visualization (depth vs performance)
- Model stability analysis using different random seeds

### Key Learnings

- Decision trees are powerful but high-variance models
- Unconstrained trees overfit aggressively
- Pipelines are essential to avoid data leakage
- Feature importance in trees is heuristic, not causal
- Single trees are rarely used directly in production and motivate ensemble methods

## Day 10 – Random Forests

### Topics Covered

- Limitations of single decision trees in real-world scenarios
- Ensemble learning and bagging intuition
- Bootstrap sampling and feature randomness
- Random Forest algorithm for regression
- Bias–variance behavior of Random Forests
- Out-of-Bag (OOB) error estimation
- Random Forest hyperparameters and their impact
- Model evaluation using R² and residual analysis
- Feature importance using tree-based methods
- Permutation feature importance for robust interpretation
- Model performance visualization:
--- Actual vs Predicted plots
--- Residual plots

### Key Learnings

- Single decision trees suffer from high variance, making them unstable in practice.
- Random Forests reduce variance by averaging multiple de-correlated trees.
- Bootstrap sampling and feature randomness are critical to improving generalization.
- Random Forests provide strong performance on nonlinear tabular data with minimal feature engineering.
- Out-of-Bag (OOB) score offers a fast and reliable internal validation method.
- Hyperparameters like min_samples_leaf and max_features play a crucial role in controlling overfitting.
- Visualization of predictions and residuals is essential for diagnosing model behavior.
- Default feature importance can be misleading; permutation importance gives more trustworthy insights.
- Random Forests are robust, production-friendly baselines but are better suited for prediction than explanation.

## Day 11 – Ensemble Learning: Bagging & Boosting (California Housing Dataset)

### Topics Covered

- California Housing dataset overview
- Feature–target separation (X and y)
- Train–test split for model validation
- Decision Tree Regressor fundamentals
- Random Forest Regressor (Bagging)
- Gradient Boosting Regressor (Boosting)
- Squared loss and residual learning
- Feature importance analysis
- Model evaluation using RMSE

### Key Learnings

- Decision Trees are highly flexible but prone to overfitting
- Bagging (Random Forest) reduces variance by averaging multiple models
- Boosting (Gradient Boosting) reduces bias by learning from previous errors
- Gradient Boosting builds models sequentially using residuals
- Squared loss emphasizes larger errors more strongly
- Ensemble methods outperform single models on real-world tabular data
- Median Income is the strongest predictor of house prices in California

## Tools Used

- Python 3.10
- Jupyter Notebook
- Conda environment

## How to Run

1. Activate conda environment
2. Open Jupyter Notebook or JupyterLab
3. Run the notebooks cell by cell
