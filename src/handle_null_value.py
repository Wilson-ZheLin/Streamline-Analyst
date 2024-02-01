import time
from util import read_file, separate_fill_null_list, contain_null_attributes_info

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
        df = fill_with_NaN(df, new_category_list)
    if interpolation_list:
        df = fill_with_interpolation(df, interpolation_list)
    return df

def fill_with_mean(df, attributes):
    for attr in attributes:
        if attr in df.columns:
            df[attr].fillna(df[attr].mean(), inplace=True)
    return df

def fill_with_median(df, attributes):
    for attr in attributes:
        if attr in df.columns:
            df[attr].fillna(df[attr].median(), inplace=True)
    return df

def fill_with_mode(df, attributes):
    for attr in attributes:
        if attr in df.columns:
            mode_value = df[attr].mode()[0] if not df[attr].mode().empty else None
            if mode_value:
                df[attr].fillna(mode_value, inplace=True)
    return df

def fill_with_interpolation(df, attributes, method='linear'):
    # method: default is 'linear'. 'time', 'index', 'pad', 'nearest', 'quadratic', 'cubic', etc.
    for attr in attributes:
        if attr in df.columns:
            df[attr] = df[attr].interpolate(method=method)
    return df

def fill_with_NaN(df, attributes):
    for attr in attributes:
        if attr in df.columns:
            df[attr].fillna('NaN', inplace=True)
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