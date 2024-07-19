import streamlit as st
import json
from openai import OpenAI
import pandas as pd
from util import developer_info, developer_info_static
from src.plot import correlation_matrix_plotly, plot_residuals, plot_predictions_vs_actual, plot_qq_plot
from src.handle_null_value import contains_missing_value, remove_high_null, fill_null_values
from src.preprocess import (
    convert_to_numeric,
    remove_rows_with_empty_target,
    remove_duplicates,
    transform_data_for_clustering,
)
from src.llm_service import (
    decide_fill_null,
    decide_encode_type,
    decide_target_attribute,
    decide_test_ratio,
    decide_regression_model,
)
from src.pca import decide_pca, perform_PCA_for_regression
from src.model_service import split_data, save_model, calculate_r2_score, calculate_mse_and_rmse, calculate_mae
from src.regression_model import train_selected_regression_model
from src.util import (
    select_Y,
    contain_null_attributes_info,
    separate_fill_null_list,
    check_all_columns_numeric,
    non_numeric_columns_and_head,
    separate_decode_list,
    get_data_overview,
    attribute_info,
    get_regression_method_name,
)


def start_training_model():
    st.session_state["start_training"] = True


def regression_model_pipeline(DF, API_KEY, GPT_MODEL, QUESTION=""):
    # XỬ LÝ CÂU HỎI CỦA USER
    client = OpenAI(
        api_key=API_KEY,
    )
    PROMPT = f'Tôi có 1 tập dữ liệu với các thuộc tính như sau: {st.session_state.DF_uploaded.columns.tolist()}.\nTôi có 1 câu hỏi từ người dùng: "{QUESTION}".\nDựa vào các thuộc tính của tập dữ liệu và câu hỏi của người dùng, hãy giúp tôi xác định target variable, các attributes và các value để thực hiện inference.\nCấu trúc file JSON:\n'
    PROMPT += '''
    {
        "target_variable": "target_variable",
        "attributes": ["attribute1", "attribute2", ...],
        "values":
        {
            "attribute1": value1,
            "attribute2": value2,
            ...
        }
    }
    '''
    messages = [
            {"role": "system", "content": "Bạn là một chuyên gia phân tích dữ liệu."},
            {"role": "user", "content": PROMPT},
        ]
    params = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
    response = client.chat.completions.create(**params)
    analysis_params = response.choices[0].message.content
    analysis_params = json.loads(analysis_params)

    columns = analysis_params["attributes"]
    if analysis_params["target_variable"] not in columns:
        columns.append(analysis_params["target_variable"])
    NEW_DF = DF[columns]

    st.divider()
    st.subheader('Phân tích câu hỏi bằng GPT-4o-mini')
    analysis_str = f'''
    Biến mục tiêu: {analysis_params['target_variable']}\n
    Các thuộc tính dùng để phân tích: {analysis_params['attributes']}\n
    Các giá trị của thuộc tính: {analysis_params['values']}
    '''
    st.success(analysis_str)

    st.subheader("Data Overview")
    if "data_origin" not in st.session_state:
        st.session_state.data_origin = NEW_DF
    st.dataframe(st.session_state.data_origin.describe(), width=1200)
    attributes = st.session_state.data_origin.columns.tolist()

    # Select the target variable
    if "target_selected" not in st.session_state:
        st.session_state.target_selected = False
    st.subheader("Target Variable")
    if not st.session_state.target_selected:

        with st.spinner("AI is analyzing the data..."):
            # attributes_for_target, types_info_for_target, head_info_for_target = attribute_info(st.session_state.data_origin)
            # st.session_state.target_Y = decide_target_attribute(attributes_for_target, types_info_for_target, head_info_for_target, GPT_MODEL, API_KEY)
            st.session_state.target_Y = analysis_params["target_variable"]

        if st.session_state.target_Y != -1:
            selected_Y = st.session_state.target_Y
            st.success("Target variable has been selected by the AI!")
            st.write(f"Target attribute selected: :green[**{selected_Y}**]")
            st.session_state.target_selected = True
        else:
            st.info("AI cannot determine the target variable from the data. Please select the target variable")
            target_col1, target_col2 = st.columns([9, 1])
            with target_col1:
                selected_Y = st.selectbox(
                    label="Select the target variable to predict:",
                    options=attributes,
                    index=len(attributes) - 1,
                    label_visibility="collapsed",
                )
            with target_col2:
                if st.button("Confirm", type="primary"):
                    st.session_state.target_selected = True
        st.session_state.selected_Y = selected_Y
    else:
        if st.session_state.target_Y != -1:
            st.success("Target variable has been selected by the AI!")
        st.write(f"Target variable selected: :green[**{st.session_state.selected_Y}**]")

    if st.session_state.target_selected:

        # Data Imputation
        st.subheader("Handle and Impute Missing Values")
        if "contain_null" not in st.session_state:
            st.session_state.contain_null = contains_missing_value(st.session_state.data_origin)

        if "filled_df" not in st.session_state:
            if st.session_state.contain_null:
                with st.status("Processing **missing values** in the data...", expanded=True) as status:
                    # st.write("Filtering out high-frequency missing rows and columns...")
                    # filled_df = remove_high_null(NEW_DF)
                    # filled_df = remove_rows_with_empty_target(filled_df, st.session_state.selected_Y)
                    # st.write("Large language model analysis...")
                    # attributes, types_info, description_info = contain_null_attributes_info(filled_df)
                    # fill_result_dict = decide_fill_null(attributes, types_info, description_info, GPT_MODEL, API_KEY)
                    # st.write("Imputing missing values...")
                    # mean_list, median_list, mode_list, new_category_list, interpolation_list = separate_fill_null_list(fill_result_dict)
                    # filled_df = fill_null_values(filled_df, mean_list, median_list, mode_list, new_category_list, interpolation_list)
                    # Store the imputed DataFrame in session_state
                    filled_df = NEW_DF.dropna()
                    st.session_state.filled_df = filled_df
                    NEW_DF = filled_df
                    status.update(label="Missing value processing completed!", state="complete", expanded=False)
                st.download_button(
                    label="Download Data with Missing Values Imputed",
                    data=st.session_state.filled_df.to_csv(index=False).encode("utf-8"),
                    file_name="imputed_missing_values.csv",
                    mime="text/csv",
                )
            else:
                st.session_state.filled_df = NEW_DF
                st.success("No missing values detected. Processing skipped.")
        else:
            st.success("Missing value processing completed!")
            if st.session_state.contain_null:
                st.download_button(
                    label="Download Data with Missing Values Imputed",
                    data=st.session_state.filled_df.to_csv(index=False).encode("utf-8"),
                    file_name="imputed_missing_values.csv",
                    mime="text/csv",
                )

        # Data Encoding
        st.subheader("Process Data Encoding")
        st.caption(
            "*For considerations of processing time, **NLP features** like **TF-IDF** have not been included in the current pipeline, long text attributes may be dropped."
        )
        if "all_numeric" not in st.session_state:
            st.session_state.all_numeric = check_all_columns_numeric(st.session_state.data_origin)

        if "encoded_df" not in st.session_state:
            if not st.session_state.all_numeric:
                with st.status(
                    "Encoding non-numeric data using **numeric mapping** and **one-hot**...", expanded=True
                ) as status:
                    non_numeric_attributes, non_numeric_head = non_numeric_columns_and_head(NEW_DF)
                    # NẾU CÓ CỘT SỐ TẦNG VÀ SỐ PHÒNG NGỦ, XỬ LÍ CÁC GIÁ TRỊ 'NHIỀU HƠN 10'
                    for non_numeric_attr in non_numeric_attributes:
                        if non_numeric_attr == 'Diện tích':
                            NEW_DF['Diện tích'] = NEW_DF['Diện tích'].apply(lambda x: x.replace(' m²', '') if type(x) == str else x)
                        if non_numeric_attr == 'Số tầng':
                            NEW_DF = NEW_DF[NEW_DF['Số tầng'] != 'Nhiều hơn 10']
                        if non_numeric_attr == 'Số phòng ngủ':
                            NEW_DF = NEW_DF[NEW_DF['Số phòng ngủ'] != 'nhiều hơn 10 phòng']
                            NEW_DF['Số phòng ngủ'] = NEW_DF['Số phòng ngủ'].apply(lambda x: x.replace(' phòng', '') if type(x) == str else x)
                        if non_numeric_attr == 'Giá/m2':
                            def process_price(price):
                                if type(price) != str:
                                    return price
                                new_price = price.split(' ')
                                value = new_price[0]
                                if '.' in value:
                                    value = value.replace('.', '')
                                if ',' in value:
                                    value = value.replace(',', '.')
                                value = float(value)
                                if len(new_price) == 1:
                                    return value
                                else:
                                    currency = new_price[1]
                                if currency == 'triệu/m²':
                                    return value
                                elif currency == 'tỷ/m²':
                                    return value * 1000
                                elif currency == 'đ/m²':
                                    return value * 0.000001
                            NEW_DF['Giá/m2'] = NEW_DF['Giá/m2'].apply(lambda x: process_price(x) if type(x)==str else x)
                            NEW_DF['Giá/m2'] = NEW_DF['Giá/m2'].fillna(NEW_DF['Giá/m2'].mean())
                    # st.write("Large language model analysis...")
                    # encode_result_dict = decide_encode_type(
                    #     non_numeric_attributes, non_numeric_head, GPT_MODEL, API_KEY
                    # )
                    st.write("Encoding the data...")
                    # convert_int_cols, one_hot_cols, drop_cols = separate_decode_list(
                    #     encode_result_dict, st.session_state.selected_Y
                    # )
                    # encoded_df, mappings = convert_to_numeric(NEW_DF, convert_int_cols, one_hot_cols, drop_cols)
                    # Store the imputed DataFrame in session_state
                    if 'Diện tích' in NEW_DF.columns.tolist():
                        NEW_DF['Diện tích'] = NEW_DF['Diện tích'].astype(float)
                        NEW_DF['Diện tích'] = NEW_DF['Diện tích'].fillna(NEW_DF['Diện tích'].mean())
                    if 'Số tầng' in NEW_DF.columns.tolist():
                        NEW_DF['Số tầng'] = NEW_DF['Số tầng'].astype(int)
                    if 'Số phòng ngủ' in NEW_DF.columns.tolist():
                        NEW_DF['Số phòng ngủ'] = NEW_DF['Số phòng ngủ'].astype(int)
                    st.session_state.encoded_df = NEW_DF
                    status.update(label="Data encoding completed!", state="complete", expanded=False)
                st.download_button(
                    label="Download Encoded Data",
                    data=st.session_state.encoded_df.to_csv(index=False).encode("utf-8"),
                    file_name="encoded_data.csv",
                    mime="text/csv",
                )
            else:
                st.session_state.encoded_df = NEW_DF
                st.success("All columns are numeric. Processing skipped.")
        else:
            st.success("Data encoded completed using numeric mapping and one-hot!")
            if not st.session_state.all_numeric:
                st.download_button(
                    label="Download Encoded Data",
                    data=st.session_state.encoded_df.to_csv(index=False).encode("utf-8"),
                    file_name="encoded_data.csv",
                    mime="text/csv",
                )
        # Correlation Heatmap
        if "df_cleaned1" not in st.session_state:
            st.session_state.df_cleaned1 = NEW_DF
        st.subheader("Correlation Between Attributes")
        st.plotly_chart(correlation_matrix_plotly(st.session_state.df_cleaned1))

        # Remove duplicate entities
        st.subheader("Remove Duplicate Entities")
        if "df_cleaned2" not in st.session_state:
            # st.session_state.df_cleaned2 = remove_duplicates(st.session_state.df_cleaned1)
            st.session_state.df_cleaned2 = st.session_state.df_cleaned1
            # DF = remove_duplicates(DF)
        st.info("Duplicate rows removed.")

        # Data Transformation
        st.subheader("Data Transformation")
        if "data_transformed" not in st.session_state:
            # st.session_state.data_transformed = transform_data_for_clustering(st.session_state.df_cleaned2)
            st.session_state.data_transformed = st.session_state.df_cleaned2
        st.success("Data transformed by standardization and box-cox if applicable.")

        # PCA
        st.subheader("Principal Component Analysis")
        st.write("Deciding whether to perform PCA...")
        if "df_pca" not in st.session_state:
            # _, n_components = decide_pca(st.session_state.df_cleaned2)
            # st.session_state.df_pca = perform_PCA_for_regression(st.session_state.data_transformed, n_components, st.session_state.selected_Y)
            st.session_state.df_pca = st.session_state.data_transformed
        st.success("Completed!")

        if "start_training" not in st.session_state:
            st.session_state["start_training"] = False

        # AI decide the testing set percentage
        if "test_percentage" not in st.session_state:
            with st.spinner("Deciding testing set percentage based on data..."):
                st.session_state.test_percentage = int(
                    decide_test_ratio(st.session_state.df_pca.shape, GPT_MODEL, API_KEY) * 100
                )

        splitting_column, balance_column = st.columns(2)
        with splitting_column:
            st.subheader("Data Splitting")
            st.caption("AI recommended test percentage for the model")
            st.slider(
                "Percentage of test set",
                1,
                25,
                st.session_state.test_percentage,
                key="test_percentage",
                disabled=st.session_state["start_training"],
            )

        with balance_column:
            st.metric(label="Test Data", value=f"{st.session_state.test_percentage}%", delta=None)
            st.toggle("Class Balancing", value=False, key="to_perform_balance", disabled=True)
            st.caption("Class balancing is not applicable to regression models.")

        st.button(
            "Start Training Model",
            on_click=start_training_model,
            type="primary",
            disabled=st.session_state["start_training"],
        )

        # Model Training
        if st.session_state["start_training"]:
            with st.container():
                st.header("Modeling")
                X_train_res, Y_train_res = select_Y(st.session_state.df_pca, st.session_state.selected_Y)

                # Splitting the data
                if not st.session_state.get("data_splitted", False):
                    (
                        st.session_state.X_train,
                        st.session_state.X_test,
                        st.session_state.Y_train,
                        st.session_state.Y_test,
                    ) = split_data(X_train_res, Y_train_res, st.session_state.test_percentage / 100, 42, True)
                    st.session_state["data_splitted"] = True

                # Decide model types:
                if "decided_model" not in st.session_state:
                    st.session_state["decided_model"] = False
                if "all_set" not in st.session_state:
                    st.session_state["all_set"] = False

                if not st.session_state["decided_model"]:
                    with st.spinner("Deciding models based on data..."):
                        shape_info, _, _, description_info = get_data_overview(st.session_state.df_pca)
                        model_dict = decide_regression_model(
                            shape_info, description_info, st.session_state.selected_Y, GPT_MODEL, API_KEY
                        )
                        model_list = list(model_dict.values())
                        if "model_list" not in st.session_state:
                            st.session_state.model_list = model_list
                        st.session_state["decided_model"] = True

                # Show modeling results
                if st.session_state["decided_model"]:
                    display_results(
                        st.session_state.X_train,
                        st.session_state.X_test,
                        st.session_state.Y_train,
                        st.session_state.Y_test,
                        analysis_params
                    )
                    st.session_state["all_set"] = True

                # Download models
                if st.session_state["all_set"]:
                    download_col1, download_col2, download_col3 = st.columns(3)
                    with download_col1:
                        st.download_button(
                            label="Download Model",
                            data=st.session_state.downloadable_model1,
                            file_name=f"{st.session_state.model1_name}.joblib",
                            mime="application/octet-stream",
                        )
                    with download_col2:
                        st.download_button(
                            label="Download Model",
                            data=st.session_state.downloadable_model2,
                            file_name=f"{st.session_state.model2_name}.joblib",
                            mime="application/octet-stream",
                        )
                    with download_col3:
                        st.download_button(
                            label="Download Model",
                            data=st.session_state.downloadable_model3,
                            file_name=f"{st.session_state.model3_name}.joblib",
                            mime="application/octet-stream",
                        )

        # Footer
        st.divider()
        if "all_set" in st.session_state and st.session_state["all_set"]:
            if "has_been_set" not in st.session_state:
                st.session_state["has_been_set"] = True
                developer_info()
            else:
                developer_info_static()

def display_results(X_train, X_test, Y_train, Y_test, analysis_params):
    st.success("Models selected based on your data!")

    # Data set metrics
    data_col1, data_col2, data_col3 = st.columns(3)
    with data_col1:
        st.metric(label="Total Data", value=len(X_train) + len(X_test), delta=None)
    with data_col2:
        st.metric(label="Training Data", value=len(X_train), delta=None)
    with data_col3:
        st.metric(label="Testing Data", value=len(X_test), delta=None)

    # Model training
    model_col1, model_col2, model_col3 = st.columns(3)
    with model_col1:
        if "model1_name" not in st.session_state:
            st.session_state.model1_name = get_regression_method_name(st.session_state.model_list[0])
        st.subheader(st.session_state.model1_name)
        with st.spinner("Model training in progress..."):
            if "model1" not in st.session_state:
                st.session_state.model1 = train_selected_regression_model(
                    X_train, Y_train, st.session_state.model_list[0]
                )
                st.session_state.y_pred1 = st.session_state.model1.predict(X_test)
                st.session_state.downloadable_model1 = save_model(st.session_state.model1)
        # Model metrics
        st.write("R2 Score: ", f":green[**{calculate_r2_score(st.session_state.y_pred1, Y_test)}**]")
        st.pyplot(plot_predictions_vs_actual(st.session_state.y_pred1, Y_test))
        mse1, rmse1 = calculate_mse_and_rmse(st.session_state.y_pred1, Y_test)
        st.write("Mean Squared Error: ", f":green[**{mse1}**]")
        st.write("Root Mean Squared Error: ", f":green[**{rmse1}**]")
        st.pyplot(plot_residuals(st.session_state.y_pred1, Y_test))
        st.write("Mean Absolute Error: ", f":green[**{calculate_mae(st.session_state.y_pred1, Y_test)}**]")
        st.pyplot(plot_qq_plot(st.session_state.y_pred1, Y_test))

    with model_col2:
        if "model2_name" not in st.session_state:
            st.session_state.model2_name = get_regression_method_name(st.session_state.model_list[1])
        st.subheader(st.session_state.model2_name)
        with st.spinner("Model training in progress..."):
            if "model2" not in st.session_state:
                st.session_state.model2 = train_selected_regression_model(
                    X_train, Y_train, st.session_state.model_list[1]
                )
                st.session_state.y_pred = st.session_state.model2.predict(X_test)
                st.session_state.downloadable_model2 = save_model(st.session_state.model2)
        # Model metrics
        st.write("R2 Score: ", f":green[**{calculate_r2_score(st.session_state.y_pred, Y_test)}**]")
        st.pyplot(plot_predictions_vs_actual(st.session_state.y_pred, Y_test))
        mse2, rmse2 = calculate_mse_and_rmse(st.session_state.y_pred, Y_test)
        st.write("Mean Squared Error: ", f":green[**{mse2}**]")
        st.write("Root Mean Squared Error: ", f":green[**{rmse2}**]")
        st.pyplot(plot_residuals(st.session_state.y_pred, Y_test))
        st.write("Mean Absolute Error: ", f":green[**{calculate_mae(st.session_state.y_pred, Y_test)}**]")
        st.pyplot(plot_qq_plot(st.session_state.y_pred, Y_test))

    with model_col3:
        if "model3_name" not in st.session_state:
            st.session_state.model3_name = get_regression_method_name(st.session_state.model_list[2])
        st.subheader(st.session_state.model3_name)
        with st.spinner("Model training in progress..."):
            if "model3" not in st.session_state:
                st.session_state.model3 = train_selected_regression_model(
                    X_train, Y_train, st.session_state.model_list[2]
                )
                st.session_state.y_pred3 = st.session_state.model3.predict(X_test)
                st.session_state.downloadable_model3 = save_model(st.session_state.model3)
        # Model metrics
        st.write("R2 Score: ", f":green[**{calculate_r2_score(st.session_state.y_pred3, Y_test)}**]")
        st.pyplot(plot_predictions_vs_actual(st.session_state.y_pred3, Y_test))
        mse3, rmse3 = calculate_mse_and_rmse(st.session_state.y_pred3, Y_test)
        st.write("Mean Squared Error: ", f":green[**{mse3}**]")
        st.write("Root Mean Squared Error: ", f":green[**{rmse3}**]")
        st.pyplot(plot_residuals(st.session_state.y_pred3, Y_test))
        st.write("Mean Absolute Error: ", f":green[**{calculate_mae(st.session_state.y_pred3, Y_test)}**]")
        st.pyplot(plot_qq_plot(st.session_state.y_pred3, Y_test))

    # PREDICTION
    st.divider()
    st.header("TRẢ LỜI CÂU HỎI CỦA USER")
    values = pd.DataFrame([analysis_params['values']])
    print('values:', values)
    predicted = st.session_state.model1.predict(values)
    st.success(f"Căn nhà được cung cấp trong câu hỏi có giá trị là :green[**{predicted[0]}**] triệu đồng.")