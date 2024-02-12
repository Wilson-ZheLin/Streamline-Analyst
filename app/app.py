import time
import streamlit as st
from streamlit_lottie import st_lottie
from util import load_lottie, stream_data
from prediction_model import prediction_model_pipeline
from cluster_model import cluster_model_pipeline
from regression_model import regression_model_pipeline
from visualization import data_visualization
from src.util import read_file_from_streamlit

st.set_page_config(page_title="Streamline Analyst", page_icon=":rocket:", layout="wide")

# TITLE SECTION
with st.container():
    st.subheader("Hello there 👋")
    st.title("Welcome to Streamline Analyst!")
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
    if st.session_state.initialized:
        st.write(stream_data("This is an application for Streamline Analyst."))
        time.sleep(0.5)
        st.write(stream_data("[Github > ](https://github.com/Wilson-ZheLin/Streamline-Analyst)"))
        st.session_state.initialized = False
    else:
        st.write("This is an application for Streamline Analyst.")
        st.write("[Github > ](https://github.com/Wilson-ZheLin/Streamline-Analyst)")

# INTRO SECTION
with st.container():
    st.divider()
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("What is Streamline Analyst?")
        st.write("""
                 Streamline Analyst is an application that allows you to analyze your data.
                 - Upload your data
                 - Perform data preprocessing
                 - Perform data analysis
                 - Perform data visualization
                 """)
    with right_column:
        st_lottie_animation = load_lottie()
        st_lottie(st_lottie_animation, height=280, key="coding")

# MAIN SECTION
with st.container():
    st.divider()
    st.header("Let's Get Started")
    left_column, right_column = st.columns([6, 4])
    with left_column:
        API_KEY = st.text_input(
            "Your API Key won't be stored or shared!",
            placeholder="Enter your API key here...",
        )
        st.write("👆Your OpenAI API key:")
        uploaded_file = st.file_uploader("Choose a data file. Your data won't be stored as well!", accept_multiple_files=False, type=['csv', 'json', 'xls', 'xlsx'])
        if uploaded_file:
            if uploaded_file.getvalue():
                uploaded_file.seek(0)
                st.session_state.DF_uploaded = read_file_from_streamlit(uploaded_file)
                st.session_state.is_file_empty = False
            else:
                st.session_state.is_file_empty = True
        
    with right_column:
        SELECTED_MODEL = st.selectbox(
        'Which OpenAI model do you want to use?',
        ('GPT-4-Turbo', 'GPT-3.5-Turbo'))

        MODE = st.selectbox(
        'Select proper data analysis mode',
        ('Predictive Classification', 'Clustering Model', 'Regression Model', 'Data Visualization'))
        
        st.write(f'Model selected: :green[{SELECTED_MODEL}]')
        st.write(f'Data analysis mode: :green[{MODE}]')

    # Proceed Button
    is_proceed_enabled = uploaded_file is not None and API_KEY != "" or uploaded_file is not None and MODE == "Data Visualization"

    # Initialize the 'button_clicked' state
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False
    if st.button('Start Analysis', disabled=(not is_proceed_enabled) or st.session_state.button_clicked, type="primary"):
        st.session_state.button_clicked = True
    if "is_file_empty" in st.session_state and st.session_state.is_file_empty:
        st.caption('Your data file is empty!')

    # Start Analysis
    if st.session_state.button_clicked:
        GPT_MODEL = 4 if SELECTED_MODEL == 'GPT-4-Turbo' else 3.5
        with st.container():
            if "DF_uploaded" not in st.session_state:
                st.error("File is empty!")
            else:
                if MODE == 'Predictive Classification':
                    prediction_model_pipeline(st.session_state.DF_uploaded, API_KEY, GPT_MODEL)
                elif MODE == 'Clustering Model':
                    cluster_model_pipeline(st.session_state.DF_uploaded, API_KEY, GPT_MODEL)
                elif MODE == 'Regression Model':
                    regression_model_pipeline(st.session_state.DF_uploaded, API_KEY, GPT_MODEL)
                elif MODE == 'Data Visualization':
                    data_visualization(st.session_state.DF_uploaded)