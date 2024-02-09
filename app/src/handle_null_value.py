import time
import numpy as np
from src.util import read_file, separate_fill_null_list, contain_null_attributes_info

def contains_missing_value(df):
    return df.isnull().values.any()

def fill_null_values(df, mean_list, median_list, mode_list, new_category_list, interpolation_list):
    if mean_list:
        df = fill_with_mean(df, mean_list)
    if median_list:
        df = fill_with_median(df, median_list)
    if mode_list:
        df = fill_with_mode(df, mode_list)
    if new_category_list:
        # df = fill_with_NaN(df, new_category_list)
        df = fill_with_interpolation(df, new_category_list)
    if interpolation_list:
        df = fill_with_interpolation(df, interpolation_list)
    return df

def remove_high_null(df, threshold_row=0.5, threshold_col=0.7):
    """
    Remove rows and columns from a DataFrame where the proportion of null values
    is greater than the specified threshold.

    :param df: Pandas DataFrame to be processed.
    :param threshold_row: Proportion threshold for null values (default is 0.5 for rows).
    :param threshold_col: Proportion threshold for null values (default is 0.7 for columns).
    :return: DataFrame with high-null rows and columns removed.
    """
    # Calculate the proportion of nulls in each column
    null_prop_col = df.isnull().mean()
    cols_to_drop = null_prop_col[null_prop_col > threshold_col].index

    # Drop columns with high proportion of nulls
    df_cleaned = df.drop(columns=cols_to_drop)

    # Calculate the proportion of nulls in each row
    null_prop_row = df_cleaned.isnull().mean(axis=1)
    rows_to_drop = null_prop_row[null_prop_row > threshold_row].index

    # Drop rows with high proportion of nulls
    df_cleaned = df_cleaned.drop(index=rows_to_drop)

    return df_cleaned

def fill_with_mean(df, attributes):
    for attr in attributes:
        if attr in df.columns:
            df[attr] = df[attr].fillna(df[attr].mean())
    return df

def fill_with_median(df, attributes):
    for attr in attributes:
        if attr in df.columns:
            df[attr] = df[attr].fillna(df[attr].median())
    return df

def fill_with_mode(df, attributes):
    for attr in attributes:
        if attr in df.columns:
            mode_value = df[attr].mode()[0] if not df[attr].mode().empty else None
            if mode_value is not None:
                df[attr] = df[attr].fillna(mode_value)
    return df

def fill_with_interpolation(df, attributes, method='linear'):
    # method: default is 'linear'. 'time', 'index', 'pad', 'nearest', 'quadratic', 'cubic', etc.
    for attr in attributes:
        if attr in df.columns:
            df[attr] = df[attr].interpolate(method=method)
    return df

# Deprecated: replaced interpolation to ensure no missing values
def fill_with_NaN(df, attributes):
    for attr in attributes:
        if attr in df.columns:
            df[attr] = df[attr].fillna('NaN')
    return df

def replace_placeholders_with_nan(df):
    placeholders = ["NA", "NULL", "?", "", "NaN", "None", "N/A", "n/a", "nan", "none"]
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: np.nan if str(x).lower() in placeholders else x)
    return df

if __name__ == '__main__':
    path = '/Users/zhe/Desktop/Github/Streamline/Streamline-Analyst/src/data/test_null.csv'
    df = read_file(path)
    # print("Contain null:", contains_missing_value(df))
    # start_time = time.time()
    # attributes, types_info, description_info = contain_null_attributes_info(df)
    # time1 = time.time()
    # print("Data preprocessing time:", time1 - start_time)
    # fill_result_dict = decide_fill_null(attributes, types_info, description_info)
    # time2 = time.time()
    # print("LLM response time:", time2 - time1)
    # mean_list, median_list, mode_list, new_category_list, interpolation_list = separate_fill_null_list(fill_result_dict)
    # new_df = fill_null_values(df, mean_list, median_list, mode_list, new_category_list, interpolation_list)
    # print("Contain null:", contains_missing_value(new_df))
    # print("Fill null time:", time.time() - time2)