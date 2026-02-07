# ML-Python-Core

## Day 1 – Python for Machine Learning

### Topics covered:
- Python basics for ML (functions, list comprehension, lambda)
- NumPy fundamentals (ndarray, vectorization, broadcasting)
- Performance comparison: loops vs vectorized operations
- Pandas basics (DataFrame, data inspection, cleaning)
- Simple feature engineering

### Files

- day01_numpy_basics.ipynb  
  Covers NumPy arrays, vectorized operations, broadcasting, and matrix math.

- day01_pandas_basics.ipynb  
  Covers data loading, inspection, handling missing values, and feature creation.

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

## Tools Used

- Python 3.10
- NumPy
- Pandas
- Jupyter Notebook
- Conda environment

## How to Run

1. Activate conda environment
2. Open Jupyter Notebook or JupyterLab
3. Run the notebooks cell by cell
