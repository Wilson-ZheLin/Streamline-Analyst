import streamlit as st
import pandas as pd

DATA_ORIGIN = "data_origin"


def store_origin_data(data: pd.DataFrame) -> None:
    if DATA_ORIGIN not in st.session_state:
        st.session_state.data_origin = data


def get_origin_data() -> pd.DataFrame | None:
    if DATA_ORIGIN not in st.session_state:
        return None
    return st.session_state.data_origin
