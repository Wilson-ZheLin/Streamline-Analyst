import io
import numpy as np
import streamlit as st
from collections import Counter
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from joblib import dump
from sklearn.metrics import roc_curve, silhouette_score, calinski_harabasz_score, davies_bouldin_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

def split_data(X, Y, test_size = 0.2, random_state = 42, perform_pca = False):
    """
    Splits the dataset into training and testing sets, optionally standardizing the data if PCA is not performed.

    :param X: Feature matrix.
    :param Y: Target vector.
    :param test_size: Proportion of the dataset to include in the test split.
    :param random_state: Controls the shuffling applied to the data before applying the split.
    :param perform_pca: Has PCA been performed or not. If not, standardizes the data.
    :return: A tuple containing split and optionally transformed datasets: X_train, X_test, Y_train, Y_test.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    if not perform_pca:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, Y_train, Y_test

def check_and_balance(X, Y, balance_threshold=0.5, method=1):
    """
    Check if the dataset is imbalanced and perform oversampling if necessary using RandomOverSampler, SMOTE, or ADASYN.

    Args:
    X (DataFrame): Feature set.
    Y (Series): Target variable.
    balance_threshold (float): Threshold for class balance.
    method (int): Method for oversampling. Options are 'random', 'smote', or 'adasyn'.

    Returns:
    X_resampled, Y_resampled (DataFrame/Series): Resampled data if imbalance is detected, else original data.
    """
    try:
        # Check the distribution of the target variable
        class_distribution = Counter(Y)

        # Determine if the dataset is imbalanced
        min_class_samples = min(class_distribution.values())
        max_class_samples = max(class_distribution.values())
        is_imbalanced = min_class_samples / max_class_samples < balance_threshold

        if is_imbalanced and method != 4:
            if method == 1:
                oversampler = RandomOverSampler(random_state=0)
            elif method == 2:
                oversampler = SMOTE(random_state=0)
            elif method == 3:
                oversampler = ADASYN(random_state=0)

            X_resampled, Y_resampled = oversampler.fit_resample(X, Y)
            return X_resampled, Y_resampled
        else:
            return X, Y
    except Exception as e:
        st.error("The target attribute may be continuous. Please check the data type.")
        st.stop()
    
def estimate_optimal_clusters(df):
    """
    Estimates the optimal number of clusters for KMeans clustering using the elbow method and silhouette scores.

    :param df: DataFrame containing the dataset to cluster.
    :return: The estimated optimal number of clusters.
    """
    sse = {}
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(df)
        sse[k] = kmeans.inertia_
    
    # Find the elbow point: compute the first and second differences of the SSE
    sse_values = list(sse.values())
    first_diff = np.diff(sse_values)  # first difference
    second_diff = np.diff(first_diff)  # second difference
    knee_point = np.argmax(second_diff) + 2
    
    # find the optimal number of clusters around the knee point
    silhouette_avg_scores = {}
    for k in range(knee_point - 1, knee_point + 2):
        if k >= 2:  # make sure k is at least 2
            kmeans = KMeans(n_clusters=k, random_state=42).fit(df)
            silhouette_avg_scores[k] = silhouette_score(df, kmeans.labels_)
    
    # Find the optimal number of clusters based on the highest average silhouette score
    optimal_clusters = max(silhouette_avg_scores, key=silhouette_avg_scores.get)
    
    return optimal_clusters

def calculate_f1_score(model, X_test, Y_test, binary_classification=True):
    """
    Calculates the F1 score for the predictions made by a model on a test set.

    The function supports both binary and multi-class settings by adjusting the 'average' parameter in the f1_score calculation.

    :param model: The trained machine learning model used for predictions.
    :param X_test: The feature matrix for the test set.
    :param Y_test: The true labels for the test set.
    :param binary_classification: If True, calculates the F1 score for binary classification. Otherwise, calculates for multi-class classification using the 'macro' average.
    :return: The F1 score of the model predictions.
    """
    y_pred = model.predict(X_test)
    if binary_classification:
        f1 = f1_score(Y_test, y_pred, average='binary')
    else:
        f1 = f1_score(Y_test, y_pred, average='macro')
    return f1

def model_score(model, X_test, Y_test):
    """
    Calculate the model score for classification models.
    """
    score = model.score(X_test, Y_test)
    return score

def fpr_and_tpr(model, X_test, Y_test):
    """
    Calculate the false positive rate and true positive rate for classification models.
    """
    Y_pred = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(Y_test, Y_pred)
    return fpr, tpr

def auc(fpr, tpr):
    """
    Calculate the area under the ROC curve for classification models.
    """
    auc = metrics.auc(fpr, tpr)
    return auc

def calculate_silhouette_score(X, labels):
    """
    Calculate the silhouette score for clustering models.
    """
    return silhouette_score(X, labels)

def calculate_calinski_harabasz_score(X, labels):
    """
    Calculate the calinski harabasz score for clustering models.
    """
    return calinski_harabasz_score(X, labels)

def calculate_davies_bouldin_score(X, labels):
    """
    Calculate the davies bouldin score for clustering models.
    """
    return davies_bouldin_score(X, labels)

def gmm_predict(X, model):
    """
    Get the predicted labels for a GMM model.
    """
    labels = model.predict(X)
    return labels

def calculate_r2_score(y_pred, Y_test):
    """
    Calculate the r2 score for regression models.
    """
    r2 = r2_score(Y_test, y_pred)
    return r2

def calculate_mse_and_rmse(y_pred, Y_test):
    """
    Calculate the mean squared error and root mean squared error for regression models.
    """
    mse = mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    return mse, rmse

def calculate_mae(y_pred, Y_test):
    """
    Calculate the mean absolute error for regression models.
    """
    mae = mean_absolute_error(Y_test, y_pred)
    return mae

def save_model(model):
    """
    Serializes a machine learning model into a binary format using joblib's dump function and stores it in a BytesIO buffer.
    """
    buffer = io.BytesIO()
    dump(model, buffer)
    buffer.seek(0)
    return buffer.getvalue()