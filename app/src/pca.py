import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def decide_pca(df, cumulative_variance_threshold=0.95, min_dim_reduction_ratio=0.1):
    """
    Assess whether PCA should be performed on a given DataFrame.

    Args:
    df (pandas.DataFrame): The DataFrame to be assessed.
    cumulative_variance_threshold (float): Cumulative variance threshold for PCA.
    min_dim_reduction_ratio (float): Minimum dimension reduction ratio required to perform PCA.

    Returns:
    bool: True if PCA should be performed, False otherwise.
    int: Suggested number of components based on the threshold.
    """
    # Remove non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Standardizing the Data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    # PCA for Explained Variance
    pca = PCA()
    pca.fit(scaled_data)

    # Calculate cumulative variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find the number of components for the desired threshold
    n_components = np.where(cumulative_variance >= cumulative_variance_threshold)[0][0] + 1

    # Calculate the dimension reduction ratio
    dim_reduction_ratio = 1 - (n_components / df.shape[1])

    # Check if PCA should be performed based on the dimension reduction ratio
    perform_pca = dim_reduction_ratio >= min_dim_reduction_ratio
    return perform_pca, n_components

def perform_pca(df, n_components):
    """
    Perform PCA on a given DataFrame.
    """
    # Remove non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Standardizing the Data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    # Applying PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    
    # Create a new DataFrame with principal components
    columns = [f'PC{i+1}' for i in range(n_components)]
    return pd.DataFrame(data=principal_components, columns=columns)
