import streamlit as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

@st.cache_data
def train_selected_model(X_train, Y_train, model_type, model_params=None):
    """
    Trains a specific classification model based on the provided model type and parameters.

    Parameters:
    - X_train (array-like): The training input samples.
    - Y_train (array-like): The target labels for classification.
    - model_type (int): Specifies the type of classification model to be trained.
        1 for Logistic Regression, 2 for Support Vector Machine (SVM), 3 for Naive Bayes,
        4 for Random Forest, 5 for AdaBoost, 6 for XGBoost, and 7 for Gradient Boosting.
    - model_params (dict, optional): A dictionary of parameters for the model. Defaults to None.

    Returns:
    - model: The trained model object based on the specified type.
    """
    if model_type == 1:
        return LogisticRegression_train(X_train, Y_train, model_params)
    elif model_type == 2:
        return SVM_train(X_train, Y_train, model_params)
    elif model_type == 3:
        return NaiveBayes_train(X_train, Y_train, model_params)
    elif model_type == 4:
        return RandomForest_train(X_train, Y_train, model_params=model_params)
    elif model_type == 5:
        return AdaBoost_train(X_train, Y_train, model_params)
    elif model_type == 6:
        return XGBoost_train(X_train, Y_train, model_params)
    elif model_type == 7:
        return GradientBoosting_train(X_train, Y_train, model_params)

def LogisticRegression_train(X_train, Y_train, model_params=None):
    if model_params is None: model_params = {}
    logreg = LogisticRegression(**model_params)
    logreg.fit(X_train, Y_train)
    return logreg

def SVM_train(X_train, Y_train, model_params=None):
    if model_params is None: model_params = {}
    svm = SVC(**model_params)
    svm.fit(X_train, Y_train)
    return svm

def NaiveBayes_train(X_train, Y_train, model_params=None):
    if model_params is None: model_params = {}
    nb = GaussianNB(**model_params)
    nb.fit(X_train, Y_train)
    return nb

def RandomForest_train(X_train, Y_train, n_estimators=100, random_state=None, model_params=None):
    if model_params is None: model_params = {}
    rf_params = {'n_estimators': n_estimators, 'random_state': random_state}
    rf_params.update(model_params)
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, Y_train)
    return rf

def AdaBoost_train(X_train, Y_train, model_params=None):
    if model_params is None: model_params = {}
    ab = AdaBoostClassifier(**model_params)
    ab.fit(X_train, Y_train)
    return ab

def XGBoost_train(X_train, Y_train, model_params=None):
    if model_params is None: model_params = {}
    xgb = XGBClassifier(**model_params)
    xgb.fit(X_train, Y_train)
    return xgb

def GradientBoosting_train(X_train, Y_train, model_params=None):
    if model_params is None: model_params = {}
    gb = GradientBoostingClassifier(**model_params)
    gb.fit(X_train, Y_train)
    return gb