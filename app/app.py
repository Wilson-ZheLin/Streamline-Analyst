import time
from data_preprocessing import preprocess_pipeline
import streamlit as st
from streamlit_lottie import st_lottie
from src.data_preprocess.pipeline_selector import get_selected_pipeline
from util import load_lottie, stream_data, welcome_message, introduction_message
from prediction_model import prediction_model_pipeline
from cluster_model import cluster_model_pipeline
from regression_model import regression_model_pipeline
from src.visualization import data_visualization, preprocessing
from src.util import read_file_from_streamlit

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="Streamline Analyst", page_icon=":rocket:", layout="wide")
API_KEY = os.getenv("OPENAI_KEY")

# MAIN SECTION
with st.container():
    st.divider()
    st.header("Let's Get Started")
    left_column, right_column = st.columns([6, 4])
    with left_column:
        QUESTION = st.text_input(
            "Nhập câu hỏi phân tích của bạn",
            placeholder="...",
            value="Hiển thị giá nhà trung bình của các quận ở Hà Nội",
        )
        uploaded_file = st.file_uploader(
            "Choose a data file. Your data won't be stored as well!",
            accept_multiple_files=False,
            type=["csv", "json", "xls", "xlsx"],
        )
        if uploaded_file:
            if uploaded_file.getvalue():
                uploaded_file.seek(0)
                st.session_state.DF_uploaded = read_file_from_streamlit(uploaded_file)
                st.session_state.is_file_empty = False
            else:
                st.session_state.is_file_empty = True

    with right_column:
        SELECTED_MODEL = st.selectbox("Which OpenAI model do you want to use?", ("GPT-4o-mini", "GPT-4o"))

    # Proceed Button
    is_proceed_enabled = (
        uploaded_file is not None and API_KEY != "" or uploaded_file is not None and MODE == "Data Visualization"
    )

    # Initialize the 'button_clicked' state
    if "button_clicked" not in st.session_state:
        st.session_state.button_clicked = False
    if st.button(
        "Start Analysis", disabled=(not is_proceed_enabled) or st.session_state.button_clicked, type="primary"
    ):
        st.session_state.button_clicked = True
    if "is_file_empty" in st.session_state and st.session_state.is_file_empty:
        st.caption("Your data file is empty!")

    # Start Analysis
    if st.session_state.button_clicked:
        GPT_MODEL = 4 if SELECTED_MODEL == "GPT-4-Turbo" else "gpt-3.5-turbo-1106"
        with st.container():
            if "DF_uploaded" not in st.session_state:
                st.error("File is empty!")
            else:
                # Select pipeline to process
                MODE = get_selected_pipeline(st.session_state.DF_uploaded, QUESTION, GPT_MODEL, API_KEY)
                st.write(f"Model selected: :green[{SELECTED_MODEL}]")
                st.write(f"Data analysis mode: :green[{MODE}]")

                # Start preprocessing pipeline
                preprocess_pipeline(st.session_state.DF_uploaded, API_KEY, GPT_MODEL, QUESTION)

                # # Start selected pipeline
                # if MODE == "Predictive Classification":
                #     prediction_model_pipeline(st.session_state.DF_uploaded, API_KEY, GPT_MODEL, QUESTION)
                # elif MODE == "Clustering Model":
                #     cluster_model_pipeline(st.session_state.DF_uploaded, API_KEY, GPT_MODEL, QUESTION)
                # elif MODE == "Regression Model":
                #     regression_model_pipeline(st.session_state.DF_uploaded, API_KEY, GPT_MODEL, QUESTION)
                # elif MODE == "Data Visualization":
                #     data_visualization(st.session_state.DF_uploaded, API_KEY, GPT_MODEL, QUESTION)
