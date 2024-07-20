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
from src.preprocess import (
    ENCODE_TYPE,
    convert_to_numeric,
    convert_to_one_hot,
    encode_data,
    get_encode_col_by_encode_type,
    remove_rows_with_empty_target,
    remove_duplicates,
    transform_data_for_clustering,
)
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
    get_numeric_columns,
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

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Data Info")
        st.dataframe(get_info(data), width=600)

    with col2:
        st.subheader("Data Overview")
        st.dataframe(data.head(10), width=1200)

    st.subheader("Numeric Data Processing")
    st.caption("*For considerations of processing time, **NLP features** like **TF-IDF** have not been included in the current pipeline, long text attributes may be dropped.")  # fmt: skip

    st.session_state.all_numeric = check_all_columns_numeric(st.session_state.data_origin)
    if not st.session_state.all_numeric:
        with st.status("Processing **non-numeric values** in the data...", expanded=True) as status:
            st.warning("Non-numeric columns found!")
            st.write(non_numeric_columns_and_head(data)[0])
            st.info("AI is handling non-numeric columns!")
            encode_map = decide_encode_type(attributes=data.columns, data_frame_head=data.head(20), model_type=GPT_MODEL, user_api_key=API_KEY)  # fmt: skip
            st.session_state.data_encode_map = encode_map
            encoded_data = encode_data(data, encode_map, enable_one_hot=False)
            data = encoded_data
            status.update(label="Non-numeric values processing completed!", state="complete", expanded=False)
            st.dataframe(encoded_data.head(10), width=1200)
        st.download_button(
            label="Download converted data",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="numeric_values_processed.csv",
            mime="text/csv",
        )
    else:
        st.info("All columns are numeric! No encoding needed.")

    st.subheader("Handle Missing and Duplicated Values")
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
            st.write("Removing duplicated values...")
            data = remove_duplicates(filled_df)
            status.update(label="Missing and duplicated value processing completed!", state="complete", expanded=False)
            st.dataframe(filled_df.head(10), width=1200)
        st.download_button(
            label="Download Data with Missing Values Imputed",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="imputed_missing_values.csv",
            mime="text/csv",
        )
    else:
        st.info("No missing values found!")

    st.subheader("Process Data Transformation")

    with st.status("Processing **data transformation** ...", expanded=True) as status:
        one_hot_encode_cols = get_encode_col_by_encode_type(ENCODE_TYPE["one_hot"], st.session_state.data_encode_map)
        if one_hot_encode_cols:
            st.write("One-hot encoding columns:", one_hot_encode_cols)
            st.write("Encoding data...")
            encoded_data, _ = convert_to_one_hot(data, one_hot_encode_cols)
        else:
            st.info("No one-hot encoding needed!")
        st.write("Normalizing numeric data ...")
        transformed_data = transform_data_for_clustering(encoded_data)
        status.update(label="Data transformation completed!", state="complete", expanded=False)
        st.dataframe(transformed_data.head(10), width=1200)
        data = transformed_data
    st.download_button(
        label="Download Data after Transformation",
        data=transformed_data.to_csv(index=False).encode("utf-8"),
        file_name="transformed_data.csv",
        mime="text/csv",
    )

    #st.subheader("Feature Selection")
    #st.caption("*For considerations of processing time, **PCA** has been included in the current pipeline.")  # fmt: skip

    # st.session_state.pca, n_components = decide_pca(filled_df)
    # if st.session_state.pca:
    #     with st.status("Processing **PCA** ...", expanded=True) as status:
    #         st.info("Correlation Between Attributes Before PCA")
    #         st.plotly_chart(correlation_matrix_plotly(data[get_numeric_columns(data)]))
    #         st.write("Performing PCA...")
    #         pca_data = perform_pca(data, n_components)
    #         status.update(label="PCA completed!", state="complete", expanded=False)
    #         st.dataframe(pca_data.head(10), width=1200)
    #         st.info("Correlation Between Attributes After PCA")
    #         st.plotly_chart(correlation_matrix_plotly(pca_data))
    #     st.download_button(
    #         label="Download Data after PCA",
    #         data=pca_data.to_csv(index=False).encode("utf-8"),
    #         file_name="pca_data.csv",
    #         mime="text/csv",
    #     )
    # else:
    #     st.info("No PCA needed!")

    st.success("Data preprocessing completed!")
    st.session_state.data_preprocessed = data
