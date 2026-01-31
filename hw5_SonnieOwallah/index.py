import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RANSACRegressor, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


# loading dataset
housing_data = fetch_california_housing(as_frame=True)
print(housing_data.frame.head())
print(housing_data.frame.isnull().sum())  #checking for any missing values in the dataset columns

X, y = housing_data.data, housing_data.target
print(f"California Housing Data Shape: {X.shape}")


## plotting to show correlations btwn different features in the dataset
sns.set_context("notebook", font_scale=0.6) 
sns.pairplot(housing_data.frame, height= 1, aspect=0.8)
plt.tight_layout()
plt.show()


# splitting data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)

# standard schaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


# training Random Forest separately for residual plot
rf_model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
rf_model.fit(X_train_std, y_train)

y_train_pred_rf = rf_model.predict(X_train_std)
y_test_pred_rf = rf_model.predict(X_test_std)


# function to evaluate regressors
def main(regressor, X_train, X_test, y_train, y_test):
    start_time = time.time()
    regressor.fit(X_train, y_train)
    end_time = time.time()
    train_time = end_time - start_time
    
    y_train_pred = regressor.predict(X_train)
    y_test_pred = regressor.predict(X_test)

    # getting mean squared error & r2 score
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # print(f"{regressor} MSE train: {mse_train}")
    # print(f"{regressor} MSE test: {mse_test}")
    # print(f"{regressor} R2 score train: {r2_train}")
    # print(f"{regressor} R2 score test: {r2_test}")
    # print(f"{regressor} Training Time: {train_time}")

    
    return {
        "MSE Train": mse_train,
        "MSE Test" : mse_test,
        "R2 Train": r2_train,
        "R2 Test" : r2_test,
        "Training Time": train_time
    }


# regressors
regressors = {
    "LinearRegression": LinearRegression(),
    "RANSACRegressor": RANSACRegressor(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "ElasticNet": ElasticNet(),
    "RandomForestRegressor": RandomForestRegressor()
}

# hyperparameter dictionary for evaluating each regressor
param_grid = {
    "Ridge": {'alpha': [0.1, 1.0, 10.0]},
    "Lasso": {'alpha': [0.01, 0.1, 1.0]},
    "ElasticNet": {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.2, 0.5, 0.8]},
    "RandomForestRegressor": {"n_estimators": [50, 100], "max_depth": [None, 10, 20]}
}

# evaluating regressors
results = []
for name, regressor in regressors.items():
    if name in param_grid:
        params = param_grid[name]
        for n_estimators in params.get('n_estimators', [None]):
          for max_depth in params.get('max_depth', [None]):
            for alpha in params.get('alpha', [None]):
                for l1_ratio in params.get('l1_ratio', [None]):
                    if name == "Ridge":
                        regressor.set_params(alpha=alpha)
                    elif name == "Lasso":
                        regressor.set_params(alpha=alpha)
                    elif name == "ElasticNet":
                        regressor.set_params(alpha=alpha, l1_ratio=l1_ratio)
                    elif name == 'RandomForestRegressor':
                        regressor.set_params(n_estimators=n_estimators, max_depth=max_depth)
                    



                    result = main(regressor, X_train_std, X_test_std, y_train, y_test)
                    result["Regressor"] = name
                    result["Alpha"] = alpha
                    result["L1 Ratio"] = l1_ratio
                    result['n_estimators'] = n_estimators
                    result['max_depth'] = max_depth
                    results.append(result)
    else:
        result = main(regressor, X_train_std, X_test_std, y_train, y_test)
        result["Regressor"] = name
        result["Alpha"] = None
        result["L1 Ratio"] = None
        result['n_estimators'] = None
        result['max_depth'] = None
        results.append(result) 

# converting and printing results to a dataframe for easier analysis
results_df = pd.DataFrame(results)
print(results_df)




# plotting hyperparameter effects
def hyperparameter_plot(results_df, regressor_name):
    regressor_results = results_df[results_df["Regressor"] == regressor_name]

    # MSE plot
    plt.subplot(1, 2, 1)
    plt.plot(regressor_results["Alpha"], regressor_results["MSE Train"], marker="o", label="MSE Train")
    plt.plot(regressor_results["Alpha"], regressor_results["MSE Test"], marker="o", label="MSE Test")
    plt.xscale("log")
    plt.xlabel("Alpha")
    plt.ylabel("MSE")
    plt.title(f"Effect of Alpha on MSE ({regressor_name})")
    plt.legend()

    # R2 Plot
    plt.subplot(1, 2, 2)
    plt.plot(regressor_results["Alpha"], regressor_results["R2 Train"], marker="o", label="R2 Train")
    plt.plot(regressor_results["Alpha"], regressor_results["R2 Test"], marker="o", label="R2 Test")
    plt.xscale("log")
    plt.xlabel("Alpha")
    plt.ylabel("R2")
    plt.title(f"Effect of Alpha on R2 ({regressor_name})")
    plt.legend()

    plt.tight_layout()
    plt.show()

# random forest residual plot
def plot_residuals(y_true_train, y_pred_train, y_true_test, y_pred_test, model_name):
    residuals_train = y_true_train - y_pred_train
    residuals_test = y_true_test - y_pred_test

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Training Residuals
    sns.scatterplot(x=y_pred_train, y=residuals_train, alpha=0.6, ax=axes[0])
    axes[0].axhline(y=0, color='r', linestyle='--')  # Zero residual reference line
    axes[0].set_xlabel("Predicted Values (Train)")
    axes[0].set_ylabel("Residuals (Train)")
    axes[0].set_title(f"Training Residual Plot - {model_name}")

    # Test Residuals
    sns.scatterplot(x=y_pred_test, y=residuals_test, alpha=0.6, ax=axes[1])
    axes[1].axhline(y=0, color='r', linestyle='--')  # Zero residual reference line
    axes[1].set_xlabel("Predicted Values (Test)")
    axes[1].set_ylabel("Residuals (Test)")
    axes[1].set_title(f"Test Residual Plot - {model_name}")

    plt.tight_layout()
    plt.show()


    # Plot the effect of alpha for Ridge, Lasso, ElasticNet Regressors and Random Forest Regressor
hyperparameter_plot(results_df, "Ridge")
hyperparameter_plot(results_df, "Lasso")
hyperparameter_plot(results_df, "ElasticNet")
plot_residuals(y_train, y_train_pred_rf, y_test, y_test_pred_rf, "Random Forest Regressor")

