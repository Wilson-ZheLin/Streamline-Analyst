from util import read_file

def initial(df):
    shape = df.shape
    info = df.info()
    types = df.dtypes
    description = df.describe(include='all')
    null_info = df.isnull()
    print(shape, '\n')
    print(info, '\n')
    print(types, '\n')
    print(description, '\n')
    print(null_info.sum(), '\n')

if __name__ == '__main__':
    path = 'data/survey lung cancer.csv'
    df = read_file(path)
    initial(df)