import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.preprocess import convert_to_integer

def decide_pca(df, cumulative_variance_threshold=0.95, min_dim_reduction_ratio=0.1):
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

def perform_pca(df, n_components, Y_name):
    """
    Perform PCA on a given DataFrame.
    """
    # Save the target column data
    target_data = df[Y_name]

    # Remove non-numeric columns and the target column
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[Y_name], errors='ignore')

    # Standardizing the Data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    # Applying PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    
    # Create a new DataFrame with principal components
    columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(data=principal_components, columns=columns)

    # Reattach the target column
    pca_df[Y_name] = target_data.reset_index(drop=True)
    pca_df, _ = convert_to_integer(pca_df, columns_to_convert=[Y_name])

    return pca_df
