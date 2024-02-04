import streamlit as st
from streamlit_lottie import st_lottie
from util import load_lottie, stream_data
from prediction_model import prediction_model_pipeline
from cluster_model import cluster_model_pipeline
from visualization import data_visualization
from src.util import read_file_from_streamlit

st.set_page_config(page_title="Streamline Analyst", page_icon=":rocket:", layout="wide")

# HEADER SECTION
with st.container():
    st.subheader("Hello there 👋")
    st.title("Welcome to Streamline Analyst!")
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
    if st.session_state.initialized:
        st.write(stream_data("This is an application for Streamline Analyst."))
        st.write(stream_data("[Github > ](https://github.com/Wilson-ZheLin/Streamline-Analyst)"))
        st.session_state.initialized = False
    else:
        st.write("This is an application for Streamline Analyst.")
        st.write("[Github > ](https://github.com/Wilson-ZheLin/Streamline-Analyst)")

# CONTENT SECTION
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
    st.header("Getting Started")
    left_column, right_column = st.columns([6, 4])
    with left_column:
        API_KEY = st.text_input(
            "Your API Key won't be stored or shared!",
            placeholder="Enter your API key here...",
        )
        st.write("👆Your OpenAI API key:")
        uploaded_file = st.file_uploader("Choose a data file. Your data won't be stored as well!", accept_multiple_files=False, type=['csv', 'json', 'xls', 'xlsx'])
        if uploaded_file:
            DF = read_file_from_streamlit(uploaded_file)
        
    with right_column:
        SELECTED_MODEL = st.selectbox(
        'Which OpenAI model do you want to use?',
        ('GPT-4-Turbo', 'GPT-3.5-Turbo'))

        MODE = st.selectbox(
        'Select proper data analysis mode',
        ('Predictive Classification', 'Clustering Model', 'Data Visualization'))
        
        st.write(f'Model selected: :green[{SELECTED_MODEL}]')
        st.write(f'Data analysis mode: :green[{MODE}]')

    # Proceed Button
    is_proceed_enabled = uploaded_file is not None and API_KEY != "" or uploaded_file is not None and MODE == "Data Visualization"

    # Initialize the 'button_clicked' state
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False

    if st.button('Start Analysis', disabled=not is_proceed_enabled or st.session_state.button_clicked, type="primary"):
        st.session_state.button_clicked = True

    if st.session_state.button_clicked:
        GPT_MODEL = 4 if SELECTED_MODEL == 'GPT-4-Turbo' else 3.5
        with st.container():
            if MODE == 'Predictive Classification':
                prediction_model_pipeline(DF, API_KEY, GPT_MODEL)
            elif MODE == 'Clustering Model':
                cluster_model_pipeline(DF, API_KEY, GPT_MODEL)
            elif MODE == 'Data Visualization':
                data_visualization(DF)

