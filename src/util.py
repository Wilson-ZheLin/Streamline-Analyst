import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.model_selection import train_test_split
import os

def read_file(file_path):

    # Check the size of the file
    if os.path.getsize(file_path) > 100 * 1024 * 1024:  # 100MB in bytes
        raise ValueError("Too large file")
    
    # Extract the file extension
    file_extension = file_path.split('.')[-1]

    if file_extension == 'csv':
        # Read CSV file
        return pd.read_csv(file_path)
    elif file_extension == 'json':
        # Read JSON file
        return pd.read_json(file_path)
    elif file_extension in ['xls', 'xlsx']:
        # Read Excel file
        return pd.read_excel(file_path, engine='openpyxl')
    else:
        raise ValueError("Unsupported file format: " + file_extension)

def convert_to_numeric(df):
    """
    Convert all non-numeric columns in the DataFrame to integer type,
    and return a dictionary of mappings from original values to integers.
    """
    mappings = {}
    for column in df.columns:

        if df[column].dtype == 'object':
            # Create a mapping from unique values to integers
            unique_values = df[column].unique()
            int_to_value_map = {i: value for i, value in enumerate(unique_values)}
            mappings[column] = int_to_value_map

            # Apply the reversed mapping to the DataFrame
            value_to_int_map = {v: k for k, v in int_to_value_map.items()}
            df[column] = df[column].map(value_to_int_map)

    return df, mappings

def select_Y(df, Y_name):
    if Y_name in df.columns:
        X = df.drop(Y_name, axis=1)
        Y = df[Y_name]
        return X, Y
    else:
        return -1
    

def split_data(X, Y, test_size=0.2, random_state = 42, perform_pca = False):
    """
    Split data into training and test sets.
    """
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    print('Training data count: ', len(X_train))
    print('Testing data count: ', len(X_test))

    if not perform_pca:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, Y_train, Y_test

def check_and_balance(X, Y, balance_threshold=0.5):
    """
    Check if the dataset is imbalanced and perform oversampling if necessary.

    Args:
    X (DataFrame): Feature set.
    Y (Series): Target variable.
    balance_threshold (float): Threshold for class balance.

    Returns:
    X_resampled, Y_resampled (DataFrame/Series): Resampled data if imbalance is detected, 
    else original data.
    """
    # Check the distribution of the target variable
    class_distribution = Counter(Y)

    # Determine if the dataset is imbalanced
    min_class_samples = min(class_distribution.values())
    max_class_samples = max(class_distribution.values())
    is_imbalanced = min_class_samples / max_class_samples < balance_threshold

    if is_imbalanced:
        oversampler = RandomOverSampler(random_state=0)
        X_resampled, Y_resampled = oversampler.fit_resample(X, Y)
        print("Resampled class distribution:", Counter(Y_resampled))
        return X_resampled, Y_resampled
    else:
        print("No significant imbalance detected.")
        return X, Y