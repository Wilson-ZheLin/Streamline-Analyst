import os
import io
import pandas as pd
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

def read_file(file_path):

    # Check the size of the file
    if os.path.getsize(file_path) > 200 * 1024 * 1024:  # 200MB in bytes
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

def read_file_from_streamlit(uploaded_file):

    # Check the size of the file
    if uploaded_file.size > 200 * 1024 * 1024:  # 200MB in bytes
        raise ValueError("Too large file")

    # Extract the file extension
    file_extension = uploaded_file.name.split('.')[-1]

    if file_extension == 'csv':
        # Read CSV file
        return pd.read_csv(uploaded_file)
    elif file_extension == 'json':
        # Read JSON file
        return pd.read_json(uploaded_file)
    elif file_extension in ['xls', 'xlsx']:
        # Read Excel file
        # Use io.BytesIO to handle the binary stream
        return pd.read_excel(io.BytesIO(uploaded_file.read()), engine='openpyxl')
    else:
        raise ValueError("Unsupported file format: " + file_extension)

def select_Y(df, Y_name):
    if Y_name in df.columns:
        X = df.drop(Y_name, axis=1)
        Y = df[Y_name]
        return X, Y
    else:
        return -1

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

def non_numeric_columns_and_head(df, num_rows=20):
    """
    Identify non-numeric columns in a DataFrame and return their names and head.

    :param df: Pandas DataFrame to be examined.
    :param num_rows: Number of rows to include in the head (default is 20).
    :return: A tuple with two elements:
             1. List of column names that are not numeric (integer or float).
             2. DataFrame containing the head of the non-numeric columns.
    """
    # Identify columns that are not of numeric data type
    non_numeric_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    
    # Get the head of the non-numeric columns
    non_numeric_head = df[non_numeric_cols].head(num_rows).to_csv()
    
    return non_numeric_cols, non_numeric_head

def contain_null_attributes_info(df):
    attributes = df.columns[df.isnull().any()].tolist()
    if not attributes: return [], -1, -1

    description_info = df[attributes].describe(percentiles=[.5])
    description_info = description_info.loc[['count', 'mean', '50%', 'std']].round(2).to_csv()

    dtypes_df = df[attributes].dtypes
    types_info = "\n".join([f"{index}:{dtype}" for index, dtype in dtypes_df.items()])

    return attributes, types_info, description_info

def separate_decode_list(decided_dict, Y_name):
    convert_int_cols = [key for key, value in decided_dict.items() if value == 1]
    one_hot_cols = [key for key, value in decided_dict.items() if value == 2]
    if Y_name in one_hot_cols:
        one_hot_cols.remove(Y_name)
        convert_int_cols.append(Y_name)
    return convert_int_cols, one_hot_cols

def separate_fill_null_list(fill_null_dict):
    mean_list = [key for key, value in fill_null_dict.items() if value == 1]
    median_list = [key for key, value in fill_null_dict.items() if value == 2]
    mode_list = [key for key, value in fill_null_dict.items() if value == 3]
    new_category_list = [key for key, value in fill_null_dict.items() if value == 4]
    interpolation_list = [key for key, value in fill_null_dict.items() if value == 5]
    return mean_list, median_list, mode_list, new_category_list, interpolation_list


if __name__ == '__main__':
    pass
    # decided_dict = {"GENDER": 1, "LUNG_CANCER": 1}
    # print(separate_decode_list(decided_dict, 'LUNG_CANCER'))