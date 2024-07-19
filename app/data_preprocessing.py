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


def preprocess_pipeline(DF, API_KEY, GPT_MODEL, QUESTION=""):
    store_origin_data(DF)
    data = replace_placeholders_with_nan(get_origin_data())

    st.header("Data Preprocessing")
    st.divider()  # start a new section

    # step 1: Load data -> overview
    st.subheader("Data Overview")
    st.dataframe(data.head(10), width=1200)
    st.subheader("Data Info")
    st.dataframe(get_info(data), width=600)

    # step 2: Handle missing values -> check missing values and decide how to handle them
    st.subheader("Handle Missing Values")
    st.caption("*For considerations of processing time, **high null columns** have been removed.")  # fmt: skip
    if contains_missing_value(data):
        with st.status("Processing **missing values** in the data...", expanded=True) as status:
            st.write("Filtering out high-frequency missing rows and columns...")
            filled_df = remove_high_null(data)
            st.write("Large language model analysis...")
            attributes, types_info, description_info = contain_null_attributes_info(filled_df)
            fill_result_dict = decide_fill_null(attributes, types_info, description_info, GPT_MODEL, API_KEY)
            st.write("Imputing missing values...")
            mean_list, median_list, mode_list, new_category_list, interpolation_list = separate_fill_null_list(fill_result_dict)  # fmt: skip
            filled_df = fill_null_values(filled_df, mean_list, median_list, mode_list, new_category_list, interpolation_list)  # fmt: skip
            status.update(label="Missing value processing completed!", state="complete", expanded=False)
            data = filled_df
        st.download_button(
            label="Download Data with Missing Values Imputed",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="imputed_missing_values.csv",
            mime="text/csv",
        )
        st.dataframe(filled_df.head(10), width=1200)
    else:
        st.info("No missing values found!")

    # step 3: Data encoding -> check data type and convert to numeric
    st.subheader("Data Encoding")
    st.caption("*For considerations of processing time, **NLP features** like **TF-IDF** have not been included in the current pipeline, long text attributes may be dropped.")  # fmt: skip
    st.session_state.all_numeric = check_all_columns_numeric(st.session_state.data_origin)
    if not st.session_state.all_numeric:
        st.warning("Non-numeric columns found!")
        st.write(non_numeric_columns_and_head(data)[0])
        st.info("AI is handling non-numeric columns!")
        data = convert_to_numeric(data)
