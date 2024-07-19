from env_provider import *
import pandas as pd

from src.handle_null_value import fill_null_values, remove_high_null
from src.llm_service import decide_encode_type, decide_fill_null, select_pipeline
from src.preprocess import encode_data
from src.util import contain_null_attributes_info, separate_fill_null_list
from src.utils.dataframe_util import get_info

data = pd.read_csv("/Users/ninh.nguyen/ninhnq_1911/DA_hcmus/house-price-analyzer/dataset/VN_housing_dataset.csv")
# data = data.sample(1000)

print(data.head(20))
print(data.info())


gpt_key = os.environ.get("OPENAI_KEY")
encode_map = decide_encode_type(
    attributes=data.columns, data_frame_head=data.head(20), model_type=3, user_api_key=gpt_key
)
encoded_data = encode_data(data, encode_map, enable_one_hot=False)

print(encode_map)
print(encoded_data.head(20))

data = encoded_data

print(data.head(20))

filled_df = remove_high_null(data)
attributes, types_info, description_info = contain_null_attributes_info(filled_df)
fill_result_dict = decide_fill_null(attributes, types_info, description_info, 3, gpt_key)
print(fill_result_dict)

mean_list, median_list, mode_list, new_category_list, interpolation_list = separate_fill_null_list(fill_result_dict)  # fmt: skip
filled_df = fill_null_values(filled_df, mean_list, median_list, mode_list, new_category_list, interpolation_list)  # fmt: skip

print(filled_df.head(10))
print(filled_df.info())

filled_df.to_csv("/Users/ninh.nguyen/ninhnq_1911/DA_hcmus/house-price-analyzer/local/data/preprocess.csv", index=False)

