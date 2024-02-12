import streamlit as st
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

@st.cache_data
def train_selected_regression_model(X_train, Y_train, model_type, model_params=None):
    """
    Trains a regression model based on the specified model type and parameters.

    Parameters:
    - X_train (array-like): The training input samples.
    - Y_train (array-like): The target values (real numbers).
    - model_type (int): An integer representing the type of regression model to train. 
        1 for Linear Regression, 2 for Ridge Regression, 3 for Lasso Regression, 
        4 for Random Forest Regressor, 5 for Gradient Boosting Regressor, and 6 for ElasticNet Regression.
    - model_params (dict, optional): A dictionary of model-specific parameters. Default is None.

    Returns:
    - The trained regression model object based on the specified model type.
    """
    if model_type == 1:
        return LinearRegression_train(X_train, Y_train, model_params)
    elif model_type == 2:
        return RidgeRegression_train(X_train, Y_train, model_params)
    elif model_type == 3:
        return LassoRegression_train(X_train, Y_train, model_params)
    elif model_type == 4:
        return RandomForestRegressor_train(X_train, Y_train, model_params)
    elif model_type == 5:
        return GradientBoostingRegressor_train(X_train, Y_train, model_params)
    elif model_type == 6:
        return ElasticNetRegressor_train(X_train, Y_train, model_params)

def LinearRegression_train(X_train, Y_train, model_params=None):
    if model_params is None: model_params = {}
    lr = LinearRegression(**model_params)
    lr.fit(X_train, Y_train)
    return lr

def RidgeRegression_train(X_train, Y_train, model_params=None):
    if model_params is None: model_params = {}
    ridge = Ridge(**model_params)
    ridge.fit(X_train, Y_train)
    return ridge

def LassoRegression_train(X_train, Y_train, model_params=None):
    if model_params is None: model_params = {}
    lasso = Lasso(**model_params)
    lasso.fit(X_train, Y_train)
    return lasso

def RandomForestRegressor_train(X_train, Y_train, model_params=None):
    if model_params is None: model_params = {}
    rf = RandomForestRegressor(**model_params)
    rf.fit(X_train, Y_train)
    return rf

def GradientBoostingRegressor_train(X_train, Y_train, model_params=None):
    if model_params is None: model_params = {}
    gbr = GradientBoostingRegressor(**model_params)
    gbr.fit(X_train, Y_train)
    return gbr

def ElasticNetRegressor_train(X_train, Y_train, model_params=None):
    if model_params is None: model_params = {}
    en = ElasticNet(**model_params)
    en.fit(X_train, Y_train)
    return en