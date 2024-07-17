import os
import yaml
import json
import re
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
import pandas as pd
from src.util import get_data_overview

model_name = 'gpt-3.5-turbo-1106'


mock_question = '''
Mô tả tổng quan dữ liệu
'''

column_description = """
Ngày: Ngày mà thông tin về bất động sản được ghi nhận.
Địa chỉ: Địa chỉ chi tiết của bất động sản, bao gồm tên đường, phường/xã, quận/huyện.
Quận: Đơn vị hành chính cấp hành chính lớn hơn phường/xã, nơi bất động sản đặt tại.
Huyện: Đơn vị hành chính lớn hơn phường/xã nơi bất động sản đặt tại.
Loại hình nhà ở: Mô tả loại hình của bất động sản, ví dụ nhà ngõ, nhà mặt phố, biệt thự, chung cư, v.v.
Giấy tờ pháp lý: Trạng thái pháp lý của bất động sản, bao gồm những loại như đã có sổ đỏ, đang chờ cấp sổ, hoặc không rõ ràng.
Số tầng: Số tầng của bất động sản, bao gồm cả tầng hầm và tầng trệt nếu có.
Số phòng ngủ: Số lượng phòng ngủ trong bất động sản.
Diện tích: Diện tích tổng thể của bất động sản, tính bằng mét vuông (m²).
Dài: Chiều dài của bất động sản, tính bằng mét (m).
Rộng: Chiều rộng của bất động sản, tính bằng mét (m).
Giá/m2: Giá bán hoặc giá cho thuê trung bình của một mét vuông bất động sản, tính bằng đơn vị tiền tệ (ví dụ như VND).
"""

related_template = """
    Bạn là một data analyst.  Shape của data frame là {shape_info}.  Giá trị head(5) của data frame là: {head_info}. Giá trị nunique() của data frame là: {nunique_info}. Mô tả của data frame là: {description_info}. Mô tả các cột là: {column_description}. Câu hỏi của người dùng là: {question}. Data đã được làm sạch và tiền xử lý, các null đã được lấp đầy và mã hoá để sẵn sàng để huấn luyện mô hình máy học.  Dựa vào thông tin data cung cấp, hãy giúp tôi quyết định rằng câu hỏi có liên quan đến visualization và có thể visualization trên dữ liệu data frame giá nhà được cung cấp không. Các tùy chọn là: true:Có, false:Không.  Chỉ dữ liệu được trả về ở định dạng json mà không có bất kỳ lời giải thích hoặc nội dung nào khác. Phản hồi mẫu: {{"isRelated":true}}
"""

attrs_template = """  Bạn là một data analyst.  Shape của data frame là {shape_info}.  Giá trị head(5) của data frame là: {head_info}. Giá trị nunique() của data frame là: {nunique_info}. Mô tả của data frame là: {description_info}. Mô tả các cột là: {column_description}. Câu hỏi của người dùng là: {question}. Data đã được làm sạch và tiền xử lý, các null đã được lấp đầy và mã hoá để sẵn sàng để huấn luyện mô hình máy học. Dựa vào thông tin data cung cấp, hãy giúp tôi quyết định rằng câu hỏi có liên quan đến các trường dữ liệu nào trên dữ liệu data frame giá nhà được cung cấp. Các trường tùy chọn là: ['Ngày', 'Địa chỉ', 'Quận', 'Huyện', 'Loại hình nhà ở','Giấy tờ pháp lý', 'Số tầng', 'Số phòng ngủ', 'Diện tích', 'Dài','Rộng', 'Giá/m2'] Chỉ dữ liệu được trả về ở định dạng json mà không có bất kỳ lời giải thích hoặc nội dung nào khác. Phản hồi mẫu: {{"attrs":["Ngày", "Địa chỉ"], "reason": ""}}

"""

single_attr_chart_visualization_template = """ Bạn là một data analyst.  Shape của data frame là {shape_info}.  Giá trị head(5) của data frame là: {head_info}. Giá trị nunique() của data frame là: {nunique_info}. Mô tả của data frame là: {description_info}. Mô tả các cột là: {column_description}.Các trường cần visusalization là: {attrs}. Câu hỏi của người dùng là: {question}. Data đã được làm sạch và tiền xử lý, các null đã được lấp đầy và mã hoá để sẵn sàng để huấn luyện mô hình máy học. Dựa vào thông tin data cung cấp, hãy giúp tôi quyết định rằng nên dùng biểu đồ nào trên dữ liệu data frame giá nhà được cung cấp. Các biểu đồ tùy chọn là: ['Donut chart', 'Violin plot', 'Distribution histogram', 'Boxplot', 'Density plot', 'Strip plot', 'Distribution boxplot'] Chỉ dữ liệu được trả về ở định dạng json mà không có bất kỳ lời giải thích hoặc nội dung nào khác. Phản hồi mẫu: {{"chart":["Donut chart", "Boxplot"], "reason": ""}}"""

multiple_attr_chart_visualization_template = """ Bạn là một data analyst.  Shape của data frame là {shape_info}.  Giá trị head(5) của data frame là: {head_info}. Giá trị nunique() của data frame là: {nunique_info}. Mô tả của data frame là: {description_info}. Mô tả các cột là: {column_description}.Các trường cần visusalization là: {attrs}. Câu hỏi của người dùng là: {question}. Data đã được làm sạch và tiền xử lý, các null đã được lấp đầy và mã hoá để sẵn sàng để huấn luyện mô hình máy học. Dựa vào thông tin data cung cấp, hãy giúp tôi quyết định rằng nên dùng biểu đồ nào trên dữ liệu data frame giá nhà được cung cấp. Các biểu đồ tùy chọn là: ["Violin plot", "Boxplot", "Heatmap", "Strip plot", "Line plot", "Scatter plot"] Chỉ dữ liệu được trả về ở định dạng json mà không có bất kỳ lời giải thích hoặc nội dung nào khác. Phản hồi mẫu: {{"chart":["Violin plot", "Boxplot"], "reason": ""}}"""


def create_param(df, model, api_key, question):
    
    llm = ChatOpenAI(model_name=model, openai_api_key=api_key, temperature=0)
    shape_info, head_info, nunique_info, description_info = get_data_overview(
        df)

    return {
        "question": question,
        # "df": df,
        "llm": llm,
        "shape_info": shape_info, "head_info": head_info, "nunique_info": nunique_info, "description_info": description_info
    }


def question_llm(param, summary_prompt):
    llm = param['llm']
    llm_answer = llm([HumanMessage(content=summary_prompt)])
    if '```json' in llm_answer.content:
        match = re.search(r'```json\n(.*?)```', llm_answer.content, re.DOTALL)
        if match:
            json_str = match.group(1)
    else:
        json_str = llm_answer.content
    return json.loads(json_str)


def predict_visualization_related_question(param):

    shape_info = param['shape_info']
    head_info = param["head_info"]
    nunique_info = param["nunique_info"]
    description_info = param["description_info"]
    question = param["question"]

    related_prompt_template = PromptTemplate(input_variables=[
        "shape_info", "head_info", "nunique_info", "description_info", "question", "column_description"], template=related_template)

    related_summary_prompt = related_prompt_template.format(
        shape_info=shape_info, head_info=head_info, nunique_info=nunique_info, description_info=description_info, column_description=column_description, question=question)

    related_value = question_llm(param, related_summary_prompt)
    return related_value


def predict_visualization_related_attrs(param):
    shape_info = param['shape_info']
    head_info = param["head_info"]
    nunique_info = param["nunique_info"]
    description_info = param["description_info"]
    question = param["question"]
    attrs_prompt_template = PromptTemplate(input_variables=[
        "shape_info", "head_info", "nunique_info", "description_info", "question", "column_description"], template=attrs_template)
    attrs_summary_prompt = attrs_prompt_template.format(shape_info=shape_info, head_info=head_info, nunique_info=nunique_info,
                                                        description_info=description_info, column_description=column_description, question=question)

    attrs_value = question_llm(param, attrs_summary_prompt)
    return attrs_value


def predict_visualization_chart(param, template, attrs_value):
    shape_info = param['shape_info']
    head_info = param["head_info"]
    nunique_info = param["nunique_info"]
    description_info = param["description_info"]
    question = param["question"]
    prompt_template = PromptTemplate(input_variables=[
        "shape_info", "head_info", "nunique_info", "description_info", "question", "column_description", "attrs"], template=template)
    summary_prompt = prompt_template.format(shape_info=shape_info, head_info=head_info, nunique_info=nunique_info,
                                            description_info=description_info, column_description=column_description, attrs=attrs_value["attrs"], question=question)

    value = question_llm(param, summary_prompt)
    return value
