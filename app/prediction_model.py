import streamlit as st
from src.plot import list_all
from src.handle_null_value import contains_missing_value, remove_high_null, fill_null_values
from src.llm_service import decide_fill_null
from src.util import contain_null_attributes_info, separate_fill_null_list

def prediction_model_pipeline(DF, API_KEY, SELECTED_MODEL):
    st.subheader('Data Overview')
    st.write(DF.describe())
    st.pyplot(list_all(DF))
    st.subheader('Target Variable')
    attributes = DF.columns.tolist()
    selected_Y = st.selectbox(
        label = 'Select the target variable to predict:',
        options = attributes,
        index = len(attributes)-1
    )
    st.write(f'Attribute selected: :green[{selected_Y}]')
    
    st.subheader('Handle and Impute Missing Values')
    if 'is_filled' not in st.session_state:
        st.session_state.is_filled = False

    with st.spinner("Processing **missing values** in the data..."):
        if contains_missing_value(DF) and not st.session_state.button_clicked:
            filled_df = remove_high_null(DF)
            attributes, types_info, description_info = contain_null_attributes_info(filled_df)
            fill_result_dict = decide_fill_null(attributes, types_info, description_info)
            mean_list, median_list, mode_list, new_category_list, interpolation_list = separate_fill_null_list(fill_result_dict)
            filled_df = fill_null_values(filled_df, mean_list, median_list, mode_list, new_category_list, interpolation_list)
            st.session_state.button_clicked = True
    st.success(':green[Missing value processing completed!]')

    if st.session_state.is_filled:
        st.download_button(
            label="Download Data with Missing Values Imputed",
            data=filled_df.to_csv(),
            file_name="missing_values_imputed.csv",
            mime='text/csv'
        )
        
    st.divider()
    
