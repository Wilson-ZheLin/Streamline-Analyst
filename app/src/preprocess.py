import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, PowerTransformer
from src.util import read_file, non_numeric_columns_and_head, separate_decode_list

def initial(df):
    shape = df.shape
    # df.info()
    head = df.head(10)
    nunique = df.nunique()
    types = df.dtypes
    description = df.describe(include='all')
    null_info = df.isnull()
    print(shape, '\n')
    print(head, '\n')
    print(nunique, '\n')
    print(types, '\n')
    print(description, '\n')
    print(null_info.sum(), '\n')

def convert_to_numeric(df, convert_int_cols_list, one_hot_cols_list, drop_cols):
    df, int_mapping = convert_to_integer(df, convert_int_cols_list)
    df, one_hot_mapping = convert_to_one_hot(df, one_hot_cols_list)
    df = df.drop(columns=drop_cols, errors='ignore')
    mappings = {'integer_mappings': int_mapping, 'one_hot_mappings': one_hot_mapping}
    return df, mappings

def convert_to_integer(df, columns_to_convert=[]):
    """
    Convert specified non-numeric columns in the DataFrame to integer type,
    and return a dictionary of mappings from original values to integers.

    :param df: Pandas DataFrame to be processed.
    :param columns_to_convert: List of column names to be converted to integer type.
    :return: A tuple with two elements:
             1. DataFrame with specified columns converted to integer type.
             2. Dictionary of mappings for each converted column.
    """
    mappings = {}
    for column in columns_to_convert:

        if df[column].dtype == 'object':
            # Create a mapping from unique values to integers
            unique_values = df[column].unique()
            int_to_value_map = {i: value for i, value in enumerate(unique_values)}
            mappings[column] = int_to_value_map

            # Apply the reversed mapping to the DataFrame
            value_to_int_map = {v: k for k, v in int_to_value_map.items()}
            df[column] = df[column].map(value_to_int_map)

    return df, mappings

def convert_to_one_hot(df, columns_to_convert=[]):
    """
    Convert specified non-numeric columns in the DataFrame to one-hot encoding,
    and return a modified DataFrame and a dictionary of mappings used for one-hot encoding.

    :param df: Pandas DataFrame to be processed.
    :param columns_to_convert: List of column names to be converted to one-hot encoding.
    :return: A tuple with two elements:
             1. DataFrame with specified columns converted to one-hot encoding.
             2. Dictionary of mappings for each converted column.
    """
    mappings = {}
    df_modified = df.copy()

    for column in columns_to_convert:
        # Check if the column is categorical
        if df[column].dtype == 'object' or df[column].dtype == 'category':
            # Perform one-hot encoding
            one_hot = pd.get_dummies(df[column], prefix=column)
            # Add the new columns to the modified DataFrame
            df_modified = pd.concat([df_modified, one_hot], axis=1)
            # Drop the original column
            df_modified.drop(column, axis=1, inplace=True)

            # Store the mapping
            mappings[column] = {i: column + '_' + str(i) for i in df[column].unique()}

    return df_modified, mappings

def remove_rows_with_empty_target(df, Y_name):
    """
    Remove rows from the DataFrame where the target column has empty values.

    :param df: Pandas DataFrame to be processed.
    :param Y_name: Name of the target column to check for empty values.
    :return: DataFrame with rows removed where target column value is empty.
    """
    # Remove rows where the target column is empty (NaN)
    cleaned_df = df.dropna(subset=[Y_name])
    return cleaned_df

def remove_duplicates(df):
    return df.drop_duplicates()

def transform_data_for_clustering(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    transformed_df = df.copy()
    pt = PowerTransformer(method='box-cox', standardize=False)

    for col in numeric_cols:
        if (transformed_df[col] > 0).all():
            skewness = stats.skew(transformed_df[col])
            if abs(skewness) > 0.5:
                transformed_data = pt.fit_transform(transformed_df[[col]])
                transformed_df[col] = transformed_data

    scaler = StandardScaler()
    transformed_df[numeric_cols] = scaler.fit_transform(transformed_df[numeric_cols])
    
    return transformed_df


if __name__ == '__main__':
    path = '/Users/zhe/Desktop/Github/Streamline/Streamline-Analyst/src/data/survey lung cancer.csv'
    df = read_file(path)
    # df, mappings = convert_to_numeric(df, ['GENDER', 'LUNG_CANCER'])
    # df, mappings = convert_to_one_hot(df, ['GENDER', 'LUNG_CANCER'])
    initial(df)
    # non_numeric_cols, non_numeric_head = non_numeric_columns_and_head(df)
    # print(df.head(10))
    # print(mappings)
    # print(non_numeric_cols)
    # print(non_numeric_head)

    # non_numeric_attributes, non_numeric_head = non_numeric_columns_and_head(df)
    # encode_result_dict = decide_encode_type(non_numeric_attributes, non_numeric_head)
    # convert_int_cols, one_hot_cols = separate_decode_list(encode_result_dict, "LUNG_CANCER")
    # new_df, mappings = convert_to_numeric(df, convert_int_cols, one_hot_cols)
    # print(mappings)
    # print(new_df)
    