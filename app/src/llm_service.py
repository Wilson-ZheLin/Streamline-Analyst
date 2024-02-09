import os
import yaml
import json
import re
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI

config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
model4_name = config["model4_name"]
model3_name = config["model3_name"]
api_key = config["openai_api_key"]

def decide_encode_type(attributes, data_frame_head, model_type = 4, user_api_key = None):
    try:
        model_name = model4_name if model_type == 4 else model3_name
        user_api_key = api_key if user_api_key is None else user_api_key
        llm = ChatOpenAI(model_name=model_name, openai_api_key=user_api_key, temperature=0)
        
        template = config["numeric_attribute_template"]
        prompt_template = PromptTemplate(input_variables=["attributes", "data_frame_head"], template=template)
        summary_prompt = prompt_template.format(attributes=attributes, data_frame_head=data_frame_head)
        
        llm_answer = llm([HumanMessage(content=summary_prompt)])
        if '```json' in llm_answer.content:
            match = re.search(r'```json\n(.*?)```', llm_answer.content, re.DOTALL)
            if match: json_str = match.group(1)
        else: json_str = llm_answer.content
        return json.loads(json_str)
    except Exception as e:
        st.error("Cannot access the OpenAI API. Please check your API key or network connection.")
        st.stop()

def decide_fill_null(attributes, types_info, description_info, model_type = 4, user_api_key = None):
    try:
        model_name = model4_name if model_type == 4 else model3_name
        user_api_key = api_key if user_api_key is None else user_api_key
        llm = ChatOpenAI(model_name=model_name, openai_api_key=user_api_key, temperature=0)
        
        template = config["null_attribute_template"]
        prompt_template = PromptTemplate(input_variables=["attributes", "types_info", "description_info"], template=template)
        summary_prompt = prompt_template.format(attributes=attributes, types_info=types_info, description_info=description_info)
        
        llm_answer = llm([HumanMessage(content=summary_prompt)])
        if '```json' in llm_answer.content:
            match = re.search(r'```json\n(.*?)```', llm_answer.content, re.DOTALL)
            if match: json_str = match.group(1)
        else: json_str = llm_answer.content
        return json.loads(json_str)
    except Exception as e:
        st.error("Cannot access the OpenAI API. Please check your API key or network connection.")
        st.stop()

def decide_model(shape_info, head_info, nunique_info, description_info, model_type = 4, user_api_key = None):
    try:
        model_name = model4_name if model_type == 4 else model3_name
        user_api_key = api_key if user_api_key is None else user_api_key
        llm = ChatOpenAI(model_name=model_name, openai_api_key=user_api_key, temperature=0)

        template = config["decide_model_template"]
        prompt_template = PromptTemplate(input_variables=["shape_info", "head_info", "nunique_info", "description_info"], template=template)
        summary_prompt = prompt_template.format(shape_info=shape_info, head_info=head_info, nunique_info=nunique_info, description_info=description_info)

        llm_answer = llm([HumanMessage(content=summary_prompt)])
        if '```json' in llm_answer.content:
            match = re.search(r'```json\n(.*?)```', llm_answer.content, re.DOTALL)
            if match: json_str = match.group(1)
        else: json_str = llm_answer.content
        return json.loads(json_str)
    except Exception as e:
        st.error("Cannot access the OpenAI API. Please check your API key or network connection.")
        st.stop()

def decide_cluster_model(shape_info, description_info, cluster_info, model_type = 4, user_api_key = None):
    try:
        model_name = model4_name if model_type == 4 else model3_name
        user_api_key = api_key if user_api_key is None else user_api_key
        llm = ChatOpenAI(model_name=model_name, openai_api_key=user_api_key, temperature=0)

        template = config["decide_clustering_model_template"]
        prompt_template = PromptTemplate(input_variables=["shape_info", "description_info", "cluster_info"], template=template)
        summary_prompt = prompt_template.format(shape_info=shape_info, description_info=description_info, cluster_info=cluster_info)

        llm_answer = llm([HumanMessage(content=summary_prompt)])
        if '```json' in llm_answer.content:
            match = re.search(r'```json\n(.*?)```', llm_answer.content, re.DOTALL)
            if match: json_str = match.group(1)
        else: json_str = llm_answer.content
        return json.loads(json_str)
    except Exception as e:
        st.error("Cannot access the OpenAI API. Please check your API key or network connection.")
        st.stop()

def decide_regression_model(shape_info, description_info, Y_name, model_type = 4, user_api_key = None):
    try:
        model_name = model4_name if model_type == 4 else model3_name
        user_api_key = api_key if user_api_key is None else user_api_key
        llm = ChatOpenAI(model_name=model_name, openai_api_key=user_api_key, temperature=0)

        template = config["decide_regression_model_template"]
        prompt_template = PromptTemplate(input_variables=["shape_info", "description_info", "Y_name"], template=template)
        summary_prompt = prompt_template.format(shape_info=shape_info, description_info=description_info, Y_name=Y_name)

        llm_answer = llm([HumanMessage(content=summary_prompt)])
        if '```json' in llm_answer.content:
            match = re.search(r'```json\n(.*?)```', llm_answer.content, re.DOTALL)
            if match: json_str = match.group(1)
        else: json_str = llm_answer.content
        return json.loads(json_str)
    except Exception as e:
        st.error("Cannot access the OpenAI API. Please check your API key or network connection.")
        st.stop()

def decide_target_attribute(attributes, types_info, head_info, model_type = 4, user_api_key = None):
    try:
        model_name = model4_name if model_type == 4 else model3_name
        user_api_key = api_key if user_api_key is None else user_api_key
        llm = ChatOpenAI(model_name=model_name, openai_api_key=user_api_key, temperature=0)

        template = config["decide_target_attribute_template"]
        prompt_template = PromptTemplate(input_variables=["attributes", "types_info", "head_info"], template=template)
        summary_prompt = prompt_template.format(attributes=attributes, types_info=types_info, head_info=head_info)

        llm_answer = llm([HumanMessage(content=summary_prompt)])
        if '```json' in llm_answer.content:
            match = re.search(r'```json\n(.*?)```', llm_answer.content, re.DOTALL)
            if match: json_str = match.group(1)
        else: json_str = llm_answer.content
        return json.loads(json_str)["target"]
    except Exception as e:
        st.error("Cannot access the OpenAI API. Please check your API key or network connection.")
        st.stop()

def decide_test_ratio(shape_info, model_type = 4, user_api_key = None):
    try:
        model_name = model4_name if model_type == 4 else model3_name
        user_api_key = api_key if user_api_key is None else user_api_key
        llm = ChatOpenAI(model_name=model_name, openai_api_key=user_api_key, temperature=0)

        template = config["decide_test_ratio_template"]
        prompt_template = PromptTemplate(input_variables=["shape_info"], template=template)
        summary_prompt = prompt_template.format(shape_info=shape_info)

        llm_answer = llm([HumanMessage(content=summary_prompt)])
        if '```json' in llm_answer.content:
            match = re.search(r'```json\n(.*?)```', llm_answer.content, re.DOTALL)
            if match: json_str = match.group(1)
        else: json_str = llm_answer.content
        return json.loads(json_str)["test_ratio"]
    except Exception as e:
        st.error("Cannot access the OpenAI API. Please check your API key or network connection.")
        st.stop()

def decide_balance(shape_info, description_info, balance_info, model_type = 4, user_api_key = None):
    try:
        model_name = model4_name if model_type == 4 else model3_name
        user_api_key = api_key if user_api_key is None else user_api_key
        llm = ChatOpenAI(model_name=model_name, openai_api_key=user_api_key, temperature=0)

        template = config["decide_balance_template"]
        prompt_template = PromptTemplate(input_variables=["shape_info", "description_info", "balance_info"], template=template)
        summary_prompt = prompt_template.format(shape_info=shape_info, description_info=description_info, balance_info=balance_info)

        llm_answer = llm([HumanMessage(content=summary_prompt)])
        if '```json' in llm_answer.content:
            match = re.search(r'```json\n(.*?)```', llm_answer.content, re.DOTALL)
            if match: json_str = match.group(1)
        else: json_str = llm_answer.content
        return json.loads(json_str)["method"]
    except Exception as e:
        st.error("Cannot access the OpenAI API. Please check your API key or network connection.")
        st.stop()
