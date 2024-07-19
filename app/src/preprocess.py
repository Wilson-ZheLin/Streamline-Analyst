from typing import Dict, List
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, PowerTransformer
from src.utils.general import clean_value, find_decimal_separator, parse_float

ENCODE_TYPE = {
    "int": 1,
    "float": 2,
    "one_hot": 3,
    "drop": 4,
}


def get_encode_col_by_encode_type(encode_type: str, encode_map: Dict) -> List[str]:
    """
    Get the list of columns that have the specified encoding type.

    :param encode_type: Encoding type to filter by.
    :param encode_map: Dictionary containing mappings for encoding.
    :return: List of column names with the specified encoding type.
    """
    return [col for col, type_ in encode_map.items() if type_ == encode_type]


def encode_data(df: pd.DataFrame, encode_map: dict) -> pd.DataFrame:
    """
    Encode the DataFrame using the provided mappings.

    :param df: DataFrame to be encoded.
    :param encode_map: Dictionary containing mappings for encoding.
    :return: Encoded DataFrame.
    """
    encoded_df, _ = convert_to_numeric(
        df,
        convert_int_cols_list=get_encode_col_by_encode_type(ENCODE_TYPE["int"], encode_map),
        one_hot_cols_list=get_encode_col_by_encode_type(ENCODE_TYPE["one_hot"], encode_map),
        drop_cols=get_encode_col_by_encode_type(ENCODE_TYPE["drop"], encode_map),
        convert_float_cols_list=get_encode_col_by_encode_type(ENCODE_TYPE["float"], encode_map),
    )
    return encoded_df


def convert_to_numeric(df, convert_int_cols_list=[], one_hot_cols_list=[], drop_cols=[], convert_float_cols_list=[]):
    """
    Convert specified columns in the DataFrame to numeric formats and drop specified columns.
    Integer conversion and one-hot encoding are applied based on the provided lists of columns.
    Returns a modified DataFrame and a dictionary of mappings used for conversions.

    :param df: Pandas DataFrame to be processed.
    :param convert_int_cols_list: List of column names to be converted to integer type.
    :param one_hot_cols_list: List of column names to be converted to one-hot encoding.
    :param drop_cols: List of column names to be dropped from the DataFrame.
    :return: A tuple with two elements:
             1. DataFrame with specified columns converted and specified columns dropped.
             2. Dictionary of mappings for each conversion type ('integer_mappings' and 'one_hot_mappings').
    """
    df, int_mapping = convert_to_integer(df, convert_int_cols_list)
    df, float_mapping = convert_to_float(df, convert_float_cols_list)
    df, one_hot_mapping = convert_to_one_hot(df, one_hot_cols_list)
    df = df.drop(columns=drop_cols, errors="ignore")
    mappings = {"integer_mappings": int_mapping, "one_hot_mappings": one_hot_mapping, "float_mappings": float_mapping}
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

        if df[column].dtype == "object":
            # Create a mapping from unique values to integers
            unique_values = df[column].unique()
            int_to_value_map = {i: value for i, value in enumerate(unique_values)}
            mappings[column] = int_to_value_map

            # Apply the reversed mapping to the DataFrame
            value_to_int_map = {v: k for k, v in int_to_value_map.items()}
            df[column] = df[column].map(value_to_int_map)

    return df, mappings


def convert_to_float(df, columns_to_convert=[]):
    """
    Convert specified columns in the DataFrame to float type using a generic float parser,
    and return a dictionary of mappings from original values to floats.

    :param df: Pandas DataFrame to be processed.
    :param columns_to_convert: List of column names to be converted to float type.
    :return: A tuple with two elements:
             1. DataFrame with specified columns converted to float type.
             2. Dictionary of mappings for each converted column.
    """
    mappings = {}
    for column in columns_to_convert:
        if df[column].dtype == "object":
            # Clean values
            df[column] = df[column].apply(clean_value)
            # Create a mapping from unique values to floats
            unique_values = df[column].unique()
            decimal_separator = find_decimal_separator(df, [column])
            value_to_float_map = {value: parse_float(value, decimal_separator) for value in unique_values}
            mappings[column] = value_to_float_map
            # Apply the mapping to the DataFrame
            df[column] = df[column].map(value_to_float_map)

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
        if df[column].dtype == "object" or df[column].dtype == "category":
            # Perform one-hot encoding
            one_hot = pd.get_dummies(df[column], prefix=column)
            # Add the new columns to the modified DataFrame
            df_modified = pd.concat([df_modified, one_hot], axis=1)
            # Drop the original column
            df_modified = df_modified.drop(column, axis=1)

            # Store the mapping
            mappings[column] = {i: column + "_" + str(i) for i in df[column].unique()}

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
    """
    Remove duplicate rows from the DataFrame.
    """
    return df.drop_duplicates()


def transform_data_for_clustering(df):
    """
    Transform numeric columns in the DataFrame for clustering.
    Applies a PowerTransformer to columns with skewness over a threshold and standardizes them.
    This can help in making the clustering algorithm more effective by normalizing the scale of numerical features.

    :param df: Pandas DataFrame to be transformed.
    :return: DataFrame with transformed numeric columns suitable for clustering.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    transformed_df = df.copy()
    pt = PowerTransformer(method="box-cox", standardize=False)

    for col in numeric_cols:
        if (transformed_df[col] > 0).all():
            skewness = stats.skew(transformed_df[col])
            if abs(skewness) > 0.5:
                transformed_data = pt.fit_transform(transformed_df[[col]])
                transformed_df[col] = transformed_data

    scaler = StandardScaler()
    transformed_df[numeric_cols] = scaler.fit_transform(transformed_df[numeric_cols])

    return transformed_df
