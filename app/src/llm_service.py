import os
import yaml
import json
import re
import time
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI
from src.util import read_file, non_numeric_columns_and_head, contain_null_attributes_info

config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
model4_name = config["model4_name"]
model3_name = config["model3_name"]
api_key = config["openai_api_key"]

def decide_encode_type(attributes, data_frame_head, model_type = 4):
    model_name = model4_name if model_type == 4 else model3_name
    llm = ChatOpenAI(model_name=model_name, openai_api_key=api_key, temperature=0)
    
    template = config["numeric_attribute_template"]
    prompt_template = PromptTemplate(input_variables=["attributes", "data_frame_head"], template=template)
    summary_prompt = prompt_template.format(attributes=attributes, data_frame_head=data_frame_head)
    
    llm_answer = llm([HumanMessage(content=summary_prompt)])
    json_str = re.search(r'```json\n(.*?)```', llm_answer.content, re.DOTALL).group(1)
    return json.loads(json_str)

def decide_fill_null(attributes, types_info, description_info, model_type = 4):
    model_name = model4_name if model_type == 4 else model3_name
    llm = ChatOpenAI(model_name=model_name, openai_api_key=api_key, temperature=0)
    
    template = config["null_attribute_template"]
    prompt_template = PromptTemplate(input_variables=["attributes", "types_info", "description_info"], template=template)
    summary_prompt = prompt_template.format(attributes=attributes, types_info=types_info, description_info=description_info)
    
    llm_answer = llm([HumanMessage(content=summary_prompt)])
    json_str = re.search(r'```json\n(.*?)```', llm_answer.content, re.DOTALL).group(1)
    return json.loads(json_str)

if __name__ == '__main__':
    pass
    # path = '/Users/zhe/Desktop/Github/Streamline/Streamline-Analyst/src/data/survey lung cancer.csv'
    # start_time = time.time()
    # df = read_file(path)
    # non_numeric_cols, non_numeric_head = non_numeric_columns_and_head(df)
    # mark_time = time.time()
    # print("Data preprocessing time:", mark_time - start_time)
    # encode_result_dict = decide_encode_type(non_numeric_cols, non_numeric_head)
    # print(encode_result_dict)
    # print("LLM response time:", time.time() - mark_time)

    path = '/Users/zhe/Desktop/Github/Streamline/Streamline-Analyst/src/data/test_null.csv'
    start_time = time.time()
    df = read_file(path)
    attributes, types_info, description_info = contain_null_attributes_info(df)
    mark_time = time.time()
    print("Data preprocessing time:", mark_time - start_time)
    fill_result_dict = decide_fill_null(attributes, types_info, description_info)
    print(fill_result_dict)
    print("LLM response time:", time.time() - mark_time)
