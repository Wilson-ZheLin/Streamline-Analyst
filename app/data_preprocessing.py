import streamlit as st
from src.utils.dataframe_util import get_info
from src.utils.session_manager import get_origin_data, store_origin_data
from util import developer_info, developer_info_static
from src.plot import confusion_metrix, roc, correlation_matrix_plotly
from src.handle_null_value import (
    contains_missing_value,
    remove_high_null,
    fill_null_values,
    replace_placeholders_with_nan,
)
from src.preprocess import convert_to_numeric, remove_rows_with_empty_target, remove_duplicates
from src.llm_service import (
    decide_fill_null,
    decide_encode_type,
    decide_model,
    decide_target_attribute,
    decide_test_ratio,
    decide_balance,
)
from src.pca import decide_pca, perform_pca
from src.model_service import split_data, check_and_balance, fpr_and_tpr, auc, save_model, calculate_f1_score
from src.predictive_model import train_selected_model
from src.util import (
    select_Y,
    contain_null_attributes_info,
    separate_fill_null_list,
    check_all_columns_numeric,
    non_numeric_columns_and_head,
    separate_decode_list,
    get_data_overview,
    get_selected_models,
    get_model_name,
    count_unique,
    attribute_info,
    get_balance_info,
    get_balance_method_name,
)


def preprocess_pipeline(DF, API_KEY, GPT_MODEL):
    store_origin_data(DF)
    origin_data = replace_placeholders_with_nan(get_origin_data())

    st.header("Data Preprocessing")
    st.divider()  # start a new section

    # step 1: Load data -> overview
    st.subheader("Data Overview")
    st.dataframe(origin_data.head(10), width=1200)
    st.subheader("Data Info")
    st.dataframe(get_info(origin_data), width=600)
    
    # step 2: Data encoding -> check data type and convert to numeric
    st.subheader("Data Encoding")
    st.caption("*For considerations of processing time, **NLP features** like **TF-IDF** have not been included in the current pipeline, long text attributes may be dropped.")
    st.session_state.all_numeric = check_all_columns_numeric(st.session_state.data_origin)
    if not st.session_state.all_numeric:
        st.warning("Non-numeric columns found!")
        st.write(non_numeric_columns_and_head(origin_data))
        st.info("AI is handling non-numeric columns!")
        data = convert_to_numeric(origin_data)

    # step 3: Check missing values
    st.subheader("Check Missing Values")
    if contains_missing_value(origin_data):
        st.warning("Missing values found!")
        st.write(contain_null_attributes_info(origin_data)[0])
        st.info("AI is handling missing values!")
        data = decide_fill_null(origin_data, API_KEY, GPT_MODEL)
    else:
        st.success("No missing values found!")
