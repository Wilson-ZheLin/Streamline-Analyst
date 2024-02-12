import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.preprocess import convert_to_integer

def decide_pca(df, cumulative_variance_threshold=0.95, min_dim_reduction_ratio=0.1):
    """
    Determines whether PCA should be performed based on cumulative variance threshold and dimension reduction ratio.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - cumulative_variance_threshold (float): The threshold of explained variance to retain. Default is 0.95.
    - min_dim_reduction_ratio (float): The minimum ratio of dimension reduction required to perform PCA. Default is 0.1.

    Returns:
    - perform_pca (bool): Whether PCA should be performed.
    - n_components (int): The number of principal components to retain.
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

def perform_pca(df, n_components, Y_name):
    """
    Performs PCA on the dataset, optionally excluding a target column, and standardizes the data.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - n_components (int): The number of principal components to retain.
    - Y_name (str, optional): The name of the target column to exclude from PCA. Default is None.

    Returns:
    - pca_df (DataFrame): DataFrame with principal components and optionally the target column.
    """
    # Save the target column data
    drop_columns = []
    if Y_name:
        target_data = df[Y_name]
        drop_columns.append(Y_name)

    # Remove non-numeric columns and the target column
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=drop_columns, errors='ignore')

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
    if Y_name:
        pca_df[Y_name] = target_data.reset_index(drop=True)
        pca_df, _ = convert_to_integer(pca_df, columns_to_convert=[Y_name])

    return pca_df

def perform_PCA_for_clustering(df, n_components):
    """
    Applies PCA transformation for clustering tasks on the given DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame to apply PCA.
    - n_components (int): The number of principal components to retain.

    Returns:
    - pca_df (DataFrame): DataFrame of the principal components.
    """
    # Applying PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df)
    
    # Create a new DataFrame with principal components
    columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(data=principal_components, columns=columns)
    
    return pca_df

def perform_PCA_for_regression(df, n_components, Y_name):
    """
    Applies PCA for regression tasks, excluding a specified target column from the transformation.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - n_components (int): The number of principal components to retain.
    - Y_name (str, optional): The name of the target column to exclude from PCA and append back after transformation. Default is None.

    Returns:
    - pca_df (DataFrame): A new DataFrame with principal components and the target column.
    """

    # Save the target column data
    drop_columns = []
    if Y_name:
        target_data = df[Y_name]
        drop_columns.append(Y_name)

    # Remove non-numeric columns and the target column
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=drop_columns, errors='ignore')

    # Applying PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(numeric_df)
    
    # Create a new DataFrame with principal components
    columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(data=principal_components, columns=columns)

    # Reattach the target column
    if Y_name:
        pca_df[Y_name] = target_data.reset_index(drop=True)
        pca_df, _ = convert_to_integer(pca_df, columns_to_convert=[Y_name])
    
    return pca_df