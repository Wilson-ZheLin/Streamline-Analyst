import os
import io
import pandas as pd

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

def check_all_columns_numeric(df):
    return df.select_dtypes(include=[int, float]).shape[1] == df.shape[1]

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

def attribute_info(df):
    attributes = df.columns.tolist()
    dtypes_df = df.dtypes
    types_info = "\n".join([f"{index}:{dtype}" for index, dtype in dtypes_df.items()])
    head_info = df.head(10).to_csv()

    return attributes, types_info, head_info

def get_data_overview(df):
    shape_info = str(df.shape)
    head_info = df.head().to_csv()
    nunique_info = df.nunique().to_csv()
    description_info = df.describe(include='all').to_csv()
    return shape_info, head_info, nunique_info, description_info

def separate_decode_list(decided_dict, Y_name):
    convert_int_cols = [key for key, value in decided_dict.items() if value == 1]
    one_hot_cols = [key for key, value in decided_dict.items() if value == 2]
    if Y_name and Y_name in one_hot_cols:
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

def get_selected_models(model_dict):
    return list(model_dict.values())

def get_model_name(model_no):
    if model_no == 1:
        return "Logistic Regression"
    elif model_no == 2:
        return "SVM"
    elif model_no == 3:
        return "Naive Bayes"
    elif model_no == 4:
        return "Random Forest"
    elif model_no == 5:
        return "ADA Boost"
    elif model_no == 6:
        return "XGBoost"
    elif model_no == 7:
        return "Grandient Boost"
    
def count_unique(df, Y):
    return df[Y].nunique()

if __name__ == '__main__':
    path = "/Users/zhe/Desktop/Github/Streamline/Streamline-Analyst/app/src/data/survey lung cancer.csv"
    df = read_file(path)
    shape_info, head_info, nunique_info, description_info = get_data_overview(df)
    print(shape_info)
    print(head_info)
    print(nunique_info)
    print(description_info)