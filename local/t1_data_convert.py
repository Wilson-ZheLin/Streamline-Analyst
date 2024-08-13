from env_provider import *
import pandas as pd

data = pd.read_csv("/Users/ninh.nguyen/ninhnq_1911/DA_hcmus/house-price-analyzer/dataset/VN_housing_dataset.csv")
data = data.sample(1000)

print(data.head(20))
print(data.info())

from src.preprocess import convert_to_numeric

data, mappings = convert_to_numeric(
    data,
    convert_int_cols_list=["Số phòng ngủ", "Số tầng"],
    one_hot_cols_list=["Quận", "Huyện", "Loại hình nhà ở", "Giấy tờ pháp lý"],
    drop_cols=["Unnamed: 0", "Ngày", "Địa chỉ"],
    convert_float_cols_list=["Giá/m2", "Rộng", "Dài", "Diện tích"],
)

print(data.head(20))
data.to_csv(
    "/Users/ninh.nguyen/ninhnq_1911/DA_hcmus/house-price-analyzer/dataset/VN_housing_dataset_cleaned.csv", index=False
)
