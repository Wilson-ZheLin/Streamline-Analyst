from env_provider import *
import pandas as pd

from src.llm_service import decide_encode_type, select_pipeline
from src.preprocess import encode_data
from src.utils.dataframe_util import get_info

data = pd.read_csv("/Users/ninh.nguyen/ninhnq_1911/DA_hcmus/house-price-analyzer/dataset/VN_housing_dataset.csv")
data = data.sample(1000)

print(data.head(20))
print(data.info())


gpt_key = os.environ.get("API_KEY")
encode_map = decide_encode_type(
    attributes=data.columns, data_frame_head=data.head(20), model_type=3, user_api_key=gpt_key
)
encoded_data = encode_data(data, encode_map)

print(encode_map)
print(encoded_data.head(20))

print(data.head(20))

selection = select_pipeline(
    shape_info=data.shape,
    description_info=get_info(data),
    head_info=data.head(10),
    question="Hiển thị giá nhà trung bình của các quận ở Hà Nội",
    model_type=3,
    user_api_key=gpt_key,
)

print(selection)

selection = select_pipeline(
    shape_info=data.shape,
    description_info=get_info(data),
    head_info=data.head(10),
    question="Hãy gom nhóm nhà theo giá và diện tích",
    model_type=3,
    user_api_key=gpt_key,
)

print(selection)


selection = select_pipeline(
    shape_info=data.shape,
    description_info=get_info(data),
    head_info=data.head(10),
    question="Hãy phân loại nhà theo tình trạng pháp lý",
    model_type=3,
    user_api_key=gpt_key,
)

print(selection)

selection = select_pipeline(
    shape_info=data.shape,
    description_info=get_info(data),
    head_info=data.head(10),
    question="Hãy dự đoán giá nhà trung bình của các quận ở Hà Nội",
    model_type=3,
    user_api_key=gpt_key,
)

print(selection)
