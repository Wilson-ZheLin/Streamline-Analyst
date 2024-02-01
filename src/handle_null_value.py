from util import read_file

def contains_missing_value(df):
    return df.isnull().values.any()

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

def contain_null_attributes_info(df):
    attributes = df.columns[df.isnull().any()].tolist()
    if not attributes: return [], -1, -1, -1

    head_info = df[attributes].head(20).to_csv()
    description_info = df[attributes].describe(include='all').to_csv()
    
    dtypes_df = df[attributes].dtypes.reset_index()
    dtypes_df.columns = ['Attribute', 'Dtype']
    types_info = dtypes_df.to_csv()

    return attributes, types_info, head_info, description_info

if __name__ == '__main__':
    path = '/Users/zhe/Desktop/Github/Streamline/Streamline-Analyst/src/data/test_null.csv'
    df = read_file(path)
    _, types_info, head_info, description_info = contain_null_attributes_info(df)
    print(types_info)
    print(head_info)
    print(description_info)