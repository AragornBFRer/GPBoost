import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import copy
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
import statsmodels.api as sm

def Sigma_func_spatial(theta, locations):
    sigma1_squared, rho = theta
    dists = cdist(locations, locations)
    return sigma1_squared * np.exp(-dists / rho) + np.eye(len(locations)) * 1e-6

def train_gpboost(X_train, y_train, initial_theta, Sigma_func, base_learner):
    gpboost = GPBoost(learning_rate=0.05, n_iterations=100, boost_type="hybrid")
    gpboost.fit(X_train, y_train, initial_theta, Sigma_func, base_learner)
    return gpboost

def predict_gpboost(gpboost, X_test, base_learner):
    return gpboost.predict(X_test, base_learner)

def train_lightgbm(X_train, y_train):
    train_data_lgb = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'learning_rate': 0.01,
        'num_leaves': 31,
        'verbose': -1,
    }
    lgb_model = lgb.train(params, train_data_lgb, num_boost_round=100)
    return lgb_model

def predict_lightgbm(lgb_model, X_test):
    return lgb_model.predict(X_test)

def train_catboost(X_train, y_train):
    cat_model = CatBoostRegressor(iterations=100, learning_rate=0.01, depth=3, verbose=0)
    cat_model.fit(X_train, y_train)
    return cat_model

def predict_catboost(cat_model, X_test):
    return cat_model.predict(X_test)


def plot_mse_comparison(mse_results):
    plt.figure(figsize=(8, 4))
    
    model_names = list(mse_results.keys())
    mse_values = list(mse_results.values())

    mse_df = pd.DataFrame({
        'Model': model_names,
        'MSE': mse_values
    })

    sns.set_theme(style="whitegrid")
    bar_plot = sns.barplot(x='MSE', y='Model', data=mse_df, palette='Blues')
    for index, value in enumerate(mse_values):
        bar_plot.text(value, index, f'{value:.2f}', va='center')

    plt.xlabel('Mean Squared Error')
    plt.title('Model Performance Comparison', fontsize=14)
    plt.xlim(0, max(mse_values) * 1.1)
    plt.tight_layout()
    plt.show()

def compare_models(n_train=500, n_test=500):
    train_spatial, test_interp_spatial, test_extrap_spatial = generate_train_test_data(
        n_train, n_test, model_type="spatial", sigma1_squared=1, sigma_squared=1, rho=0.1
    )
    y_train, X_train, Zb_train, epsilon_train, train_locations = train_spatial
    y_test_interp, X_test_interp, Zb_test_interp, epsilon_test_interp, interp_locations = test_interp_spatial
    y_test_extrap, X_test_extrap, Zb_test_extrap, epsilon_test_extrap, extrap_locations = test_extrap_spatial

    initial_theta = np.array([1.0, 0.01])  # Initial guess

    base_learner = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10)
    gpboost = train_gpboost(X_train, y_train, initial_theta, lambda theta: Sigma_func_spatial(theta, train_locations), base_learner)
    predictions_gpboost_interp = predict_gpboost(gpboost, X_test_interp, base_learner)
    predictions_gpboost_extrap = predict_gpboost(gpboost, X_test_extrap, base_learner)

    lgb_model = train_lightgbm(X_train, y_train)
    predictions_lgb_interp = predict_lightgbm(lgb_model, X_test_interp)
    predictions_lgb_extrap = predict_lightgbm(lgb_model, X_test_extrap)

    cat_model = train_catboost(X_train, y_train)
    predictions_cat_interp = predict_catboost(cat_model, X_test_interp)
    predictions_cat_extrap = predict_catboost(cat_model, X_test_extrap)

    mse_results = {
        "GPBoost_Int": mean_squared_error(y_test_interp, predictions_gpboost_interp),
        "GPBoost_Ext": mean_squared_error(y_test_extrap, predictions_gpboost_extrap),
        "LightGBM_Int": mean_squared_error(y_test_interp, predictions_lgb_interp),
        "LightGBM_Ext": mean_squared_error(y_test_extrap, predictions_lgb_extrap),
        "CatBoost_Int": mean_squared_error(y_test_interp, predictions_cat_interp),
        "CatBoost_Ext": mean_squared_error(y_test_extrap, predictions_cat_extrap),
    }

    for model_name in mse_results:
        print(f"{model_name}: {mse_results[model_name]}")

    plot_mse_comparison(mse_results)

if __name__ == "__main__":
    compare_models()
