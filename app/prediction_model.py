import streamlit as st
import pandas as pd
import time
from util import developer_info, developer_info_static
from src.plot import confusion_metrix, roc, correlation_matrix_plotly, get_confusion_metrix
from src.handle_null_value import contains_missing_value, remove_high_null, fill_null_values
from src.preprocess import convert_to_numeric, remove_rows_with_empty_target, remove_duplicates
from src.llm_service import (
    decide_fill_null,
    decide_encode_type,
    decide_model,
    decide_target_attribute,
    decide_test_ratio,
    decide_balance,
    convert_question_to_DF,
    decide_target_attribute_classi
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


def update_balance_data():
    st.session_state.balance_data = st.session_state.to_perform_balance


def start_training_model():
    st.session_state["start_training"] = True

def get_answer_bot(API_KEY, user_question):
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import OpenAI
    template = """{question}
    """
    prompt = PromptTemplate.from_template(template)
    llm = OpenAI(
        temperature=0.7,
        openai_api_key=API_KEY,
        max_tokens=-1
    )
    llm_chain = prompt | llm
    
    # print(results_str)
    # question = f'''Tôi có 1 danh sách kết quả sau khi train 1 tập dữ liệu file csv như sau
    # {user_question}
    # Hãy giúp tôi đánh giá kết quả này và đưa ra nhận xét về hiệu suất của các model đã train.
    
    # Hãy giải thích bằng tiếng Việt. Chỉ được trả lời bằng tiếng Việt thôi
    # '''
    question = f'''
    {user_question}
    Tôi có 1 danh sách kết quả sau khi train 1 tập dữ liệu file csv như trên. 
    Hãy dựa vào f1 score và accuracy, chọn ra 1 model TỐT NHẤT bằng cách trả lời bằng số 1 hoặc 2 hoặc 3.
    Chỉ được in ra số 1 hoặc 2 hoặc 3 thôi.
    '''

    final_report = llm_chain.invoke(question)
    return final_report

def inference_model(X_question, X_train, Y_train):
    # Model Inference
    st.session_state.model1 = train_selected_model(X_train, Y_train, st.session_state.model_list[0])
    predicted_label = st.session_state.model1.predict(X_question)
    return predicted_label

global X_question_full 
X_question_full  = pd.DataFrame()
def prediction_model_pipeline(DF, API_KEY, GPT_MODEL, QUESTION):
    DF = DF.drop("Địa chỉ", axis=1)
    DF = DF.drop("Ngày", axis=1)
    THE_PROMPT = QUESTION
    st.divider()
    st.subheader('Prediction model')
    # X_test_sample = {
    #         "Temperature": 16,
    #         "Humidity": 73,
    #         "Wind Speed": 9.5,	
    #         "Precipitation (%)":71,
    #         "Cloud Cover":"partly cloudy",
    #         "Atmospheric Pressure":999.44,
    #         "UV Index": 5,
    #         "Season":"Winter",
    #         "Visibility (km)": 4.6,
    #         "Location": "inland"
    # }
    # new_df = pd.DataFrame([X_test_sample])

    # if 'data_origin' not in st.session_state:
    #     st.session_state.data_origin = DF
    # st.dataframe(st.session_state.data_origin.head(20), width=1600)
    # st.dataframe(st.session_state.data_origin.describe(), width=1200)
    attributes = st.session_state.data_origin.columns.tolist()
    result_dataoverview = st.session_state.data_origin.describe()
    
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import OpenAI
    template = """{question}
    """

    prompt = PromptTemplate.from_template(template)
    llm = OpenAI(
        temperature=0.7,
        openai_api_key=API_KEY,
        max_tokens=-1
    )
    llm_chain = prompt | llm

    question = f'''
    Bạn là một nhà phân tích dữ liệu.
    Dựa vào thông tin data overview sau, hãy giúp tôi đánh giá dữ liệu và đưa ra nhận xét về dữ liệu.
    Hãy giải thích chi tiết bằng tiếng Việt. Chỉ được trả lời bằng tiếng Việt thôi.
    {result_dataoverview}
    '''

    dataoverview_report = llm_chain.invoke(question)
    #print(dataoverview_report)
    st.subheader('Nhận xét kết quả về thông tin dữ liệu')
    st.write(dataoverview_report)

    prompt = PromptTemplate.from_template(template)
    llm = OpenAI(
        temperature=0.7,
        openai_api_key=API_KEY,
        max_tokens=-1
    )
    llm_chain = prompt | llm
    
    # Select the target variable
    if 'target_selected' not in st.session_state:
        st.session_state.target_selected = False
    st.subheader('Target Variable')
    if not st.session_state.target_selected:

        with st.spinner("AI is analyzing the data..."):
            attributes_for_target, types_info_for_target, head_info_for_target = attribute_info(st.session_state.data_origin)
            st.session_state.target_Y = decide_target_attribute_classi(attributes_for_target, types_info_for_target, head_info_for_target, GPT_MODEL, API_KEY, THE_PROMPT)

        if st.session_state.target_Y != -1:
            selected_Y = st.session_state.target_Y
            st.success("Target variable has been selected by the AI!")
            st.write(f'Target attribute selected: :green[**{selected_Y}**]')
            st.session_state.target_selected = True
        else:
            st.info("AI cannot determine the target variable from the data. Please select the target variable")
            target_col1, target_col2 = st.columns([9, 1])
            with target_col1:
                selected_Y = st.selectbox(
                    label = 'Select the target variable to predict:',
                    options = attributes,
                    index = len(attributes)-1,
                    label_visibility='collapsed'
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
        st.subheader('Handle and Impute Missing Values')
        #if "contain_null" not in st.session_state:
            #st.session_state.contain_null = contains_missing_value(st.session_state.data_origin)

        if 'filled_df' not in st.session_state:
            # if st.session_state.contain_null:
            #     with st.status("Processing **missing values** in the data...", expanded=True) as status:
            #         st.write("Filtering out high-frequency missing rows and columns...")
            #         filled_df = remove_high_null(DF)
            #         filled_df = remove_rows_with_empty_target(filled_df, st.session_state.selected_Y)
            #         st.write("Large language model analysis...")
            #         attributes, types_info, description_info = contain_null_attributes_info(filled_df)
            #         fill_result_dict = decide_fill_null(attributes, types_info, description_info, GPT_MODEL, API_KEY)
            #         st.write("Imputing missing values...")
            #         mean_list, median_list, mode_list, new_category_list, interpolation_list = separate_fill_null_list(fill_result_dict)
            #         filled_df = fill_null_values(filled_df, mean_list, median_list, mode_list, new_category_list, interpolation_list)
            #         # Store the imputed DataFrame in session_state
            #         st.session_state.filled_df = filled_df
            #         DF = filled_df
            #         status.update(label='Missing value processing completed!', state="complete", expanded=False)
            #     st.download_button(
            #         label="Download Data with Missing Values Imputed",
            #         data=st.session_state.filled_df.to_csv(index=False).encode('utf-8'),
            #         file_name="imputed_missing_values.csv",
            #         mime='text/csv')
            # else:
                st.session_state.filled_df = DF
                st.success("No missing values detected. Processing skipped.")
        else:
            st.success("Missing value processing completed!")
            # if st.session_state.contain_null:
            #     st.download_button(
            #         label="Download Data with Missing Values Imputed",
            #         data=st.session_state.filled_df.to_csv(index=False).encode('utf-8'),
            #         file_name="imputed_missing_values.csv",
            #         mime='text/csv')

        # Data Encoding
        st.subheader("Process Data Encoding")
        st.caption("*For considerations of processing time, **NLP features** like **TF-IDF** have not been included in the current pipeline, long text attributes may be dropped.")
        if 'all_numeric' not in st.session_state:
            st.session_state.all_numeric = check_all_columns_numeric(st.session_state.data_origin)
        
        if 'encoded_df' not in st.session_state:
            if not st.session_state.all_numeric:
                with st.status("Encoding non-numeric data using **numeric mapping** and **one-hot**...", expanded=True) as status:
                    non_numeric_attributes, non_numeric_head = non_numeric_columns_and_head(DF)
                    st.write("Large language model analysis...")
                    encode_result_dict = decide_encode_type(non_numeric_attributes, non_numeric_head, GPT_MODEL, API_KEY)
                    st.write("Encoding the data...")
                    convert_int_cols, one_hot_cols, drop_cols = separate_decode_list(encode_result_dict, st.session_state.selected_Y)
                    encoded_df, mappings = convert_to_numeric(DF, convert_int_cols, one_hot_cols, drop_cols)
                    # Store the imputed DataFrame in session_state
                    st.session_state.encoded_df = encoded_df
                    DF = encoded_df
                    #print("DF sau khi encode: ", DF)

                    #preprocess_data_gianha(st.session_state.data_origin,'')
                    attributes_for_target, types_info_for_target, head_info_for_target = attribute_info(st.session_state.data_origin)
                    st.session_state.json_file = convert_question_to_DF(attributes_for_target, types_info_for_target, head_info_for_target, GPT_MODEL, API_KEY, THE_PROMPT)
                    #print(st.session_state.json_file)
                    data = pd.json_normalize(st.session_state.json_file)
                    data.to_csv('data.csv')
                    new_encoded_df, new_mappings  = convert_to_numeric(data, convert_int_cols, one_hot_cols, drop_cols)
                    print("Thông tin new_encoded_df: ", new_encoded_df)
                    data = new_encoded_df
                    print('Thong tin new_encoded_df:', data)

                    status.update(label='Data encoding completed!', state="complete", expanded=False)
                st.download_button(
                    label="Download Encoded Data",
                    data=st.session_state.encoded_df.to_csv(index=False).encode('utf-8'),
                    file_name="encoded_data.csv",
                    mime='text/csv')
            else:
                st.session_state.encoded_df = DF
                st.success("All columns are numeric. Processing skipped.")
        else:
            st.success("Data encoded completed using numeric mapping and one-hot!")
            if not st.session_state.all_numeric:
                st.download_button(
                    label="Download Encoded Data",
                    data=st.session_state.encoded_df.to_csv(index=False).encode('utf-8'),
                    file_name="encoded_data.csv",
                    mime='text/csv')
        
        # Correlation Heatmap
        if 'df_cleaned1' not in st.session_state:
            st.session_state.df_cleaned1 = DF
            st.session_state.df_cleaned1_question = new_encoded_df
        #st.subheader('Correlation Between Attributes')
        #st.plotly_chart(correlation_matrix_plotly(st.session_state.df_cleaned1))

        # Remove duplicate entities
        st.subheader('Remove Duplicate Entities')
        if 'df_cleaned2' not in st.session_state:
            st.session_state.df_cleaned2 = remove_duplicates(st.session_state.df_cleaned1)
            st.session_state.df_cleaned1_question = new_encoded_df
            # DF = remove_duplicates(DF)
        st.info("Duplicate rows removed.")
        
        # PCA
        st.subheader('Principal Component Analysis')
        st.write("Deciding whether to perform PCA...")
        if 'df_pca' not in st.session_state:
            to_perform_pca, n_components = decide_pca(st.session_state.df_cleaned2.drop(columns=[st.session_state.selected_Y]))
            #to_perform_pca_question, n_components_question = decide_pca(st.session_state.df_cleaned1_question.drop(columns=[st.session_state.selected_Y]))
            if 'to_perform_pca' not in st.session_state:
                st.session_state.to_perform_pca = to_perform_pca
                #st.session_state.to_perform_pca_question = to_perform_pca_question
                #st.session_state.df_pca_question = to_perform_pca(st.session_state.df_cleaned1_question, n_components, st.session_state.selected_Y)
            if st.session_state.to_perform_pca:
                st.session_state.df_pca = perform_pca(st.session_state.df_cleaned2, n_components, st.session_state.selected_Y)
                #st.session_state.df_pca_question = to_perform_pca(st.session_state.df_cleaned1_question, n_components, st.session_state.selected_Y)
                #print('df_pca_question: ', st.session_state.df_pca_question)
            else:
                st.session_state.df_pca = st.session_state.df_cleaned2
                #st.session_state.df_pca_question = st.session_state.df_cleaned1_question
        st.success("Completed!")

        # Splitting and Balancing
        if 'balance_data' not in st.session_state:
            st.session_state.balance_data = True
        if "start_training" not in st.session_state:
            st.session_state["start_training"] = False
        if 'model_trained' not in st.session_state:
            st.session_state['model_trained'] = False
        if 'is_binary' not in st.session_state:
            st.session_state['is_binary'] = count_unique(st.session_state.df_pca, st.session_state.selected_Y) == 2

        # AI decide the testing set percentage
        if 'test_percentage' not in st.session_state:
            with st.spinner("Deciding testing set percentage based on data..."):
                st.session_state.test_percentage = int(decide_test_ratio(st.session_state.df_pca.shape, GPT_MODEL, API_KEY) * 100)

        splitting_column, balance_column = st.columns(2)
        with splitting_column:
            st.subheader('Data Splitting')
            st.caption('AI recommended test percentage for the model')
            st.slider('Percentage of test set', 1, 25, st.session_state.test_percentage, key='test_percentage', disabled=st.session_state['start_training'])
        
        with balance_column:
            st.metric(label="Test Data", value=f"{st.session_state.test_percentage}%", delta=None)
            st.toggle('Class Balancing', value=st.session_state.balance_data, key='to_perform_balance', on_change=update_balance_data, disabled=st.session_state['start_training'])
            st.caption('Strategies for handling imbalanced data sets and to enhance machine learning model performance.')
            st.caption('AI will select the most appropriate method to balance the data.')
        
        st.button("Start Training Model", on_click=start_training_model, type="primary", disabled=st.session_state['start_training'])

        # Model Training
        if st.session_state['start_training']:
            with st.container():
                st.header("Modeling")
                X, Y = select_Y(st.session_state.df_pca, st.session_state.selected_Y)
                #X_question, Y_question = select_Y(st.session_state.df_pca_question, st.session_state.selected_Y)
                #X_question, Y_question = select_Y(st.session_state.df_cleaned1_question, st.session_state.selected_Y)
                #print('X la ', X)
                #print('X_question la', X_question)
                # Balancing
                if st.session_state.balance_data and "balance_method" not in st.session_state:
                    print("Balance ====================")
                    with st.spinner("AI is deciding the balance strategy for the data..."):
                        shape_info_balance, description_info_balance, balance_info_balance = get_balance_info(st.session_state.df_pca, st.session_state.selected_Y)
                        st.session_state.balance_method = int(decide_balance(shape_info_balance, description_info_balance, balance_info_balance, GPT_MODEL, API_KEY))
                        X_train_res, Y_train_res = check_and_balance(X, Y, method = st.session_state.balance_method)
                        X_train_res.to_csv('X_train_res.csv')
                else:
                    X_train_res, Y_train_res = X, Y
                    if 'balance_method' not in st.session_state:
                        st.session_state.balance_method = 4

                # Splitting the data
                if not st.session_state.get("data_splitted", False):  
                    st.session_state.X_train, st.session_state.X_test, st.session_state.Y_train, st.session_state.Y_test = split_data(X_train_res, Y_train_res, st.session_state.test_percentage / 100, 42, st.session_state.to_perform_pca)
                    st.session_state["data_splitted"] = True
                
                # Decide model types:
                if "decided_model" not in st.session_state:
                    st.session_state["decided_model"] = False
                if "all_set" not in st.session_state:
                    st.session_state["all_set"] = False
                
                if not st.session_state["decided_model"]:
                    with st.spinner("Deciding models based on data..."):
                        shape_info, head_info, nunique_info, description_info = get_data_overview(st.session_state.df_pca)
                        model_dict = decide_model(shape_info, head_info, nunique_info, description_info, GPT_MODEL, API_KEY)
                        model_list = get_selected_models(model_dict)
                        if 'model_list' not in st.session_state:
                            st.session_state.model_list = model_list
                        st.session_state["decided_model"] = True

                # Display results
                if st.session_state["decided_model"]:
                    from langchain_core.prompts import PromptTemplate
                    from langchain_openai import OpenAI
                    template = """{question}"""

                    prompt = PromptTemplate.from_template(template)
                    #llm = OpenAI(openai_api_key=API_KEY)
                    llm = OpenAI(
                            temperature=0.7,
                            openai_api_key=API_KEY,
                            max_tokens=-1
                    )
                    llm_chain = prompt | llm
                    
                    results = display_results(st.session_state.X_train, st.session_state.X_test, st.session_state.Y_train, st.session_state.Y_test)

                    for result in results:
                        result['confusion_matrix'] = result['confusion_matrix'].tolist()

                    results_str = str(results)
                    chosen_model = get_answer_bot(API_KEY, user_question=results_str)

                    print("=======================================================")
                    print("chosen_model: ", chosen_model)
                    # print(results_str)
                    st.subheader('Kết luận chung:')
                    question = f'''
                    Bạn là một nhà phân tích dữ liệu.
                    Tôi có 1 danh sách kết quả sau khi train 1 tập dữ liệu file csv như sau
                    {results_str}
                    Hãy giúp tôi đánh giá kết quả này và đưa ra nhận xét về hiệu suất của các model đã train.
                    Hãy giải thích chi tiết bằng tiếng Việt. Chỉ được trả lời bằng tiếng Việt thôi.
                    Sau đó, chọn ra một model tốt nhất và nêu ra lý do.
                    '''

                    final_report = llm_chain.invoke(question)
                    #print("final_report: ", final_report)
                    st.write(final_report)

                    # ques_VN = f'''Translate the following text into Vietnamese:
                    # {final_report}
                    # '''
                    # final_report_VN = llm_chain.invoke(ques_VN)
                    # print("final_report_VN: ", final_report_VN)
                    
                    # st.write(final_report_VN)

                    st.session_state["all_set"] = True

                 # Model Inference

                #X_question_full = new_encoded_df
                #print('X question full là', X_question_full)
                #ketqua = inference_model(data_inference,st.session_state.X_train, st.session_state.Y_train)
                #print(ketqua)
                
                st.header("Inference Model To Give The Result For The Question")

                with st.spinner("Model inference in progress..."):
                    time.sleep(5)
                    from langchain_core.prompts import PromptTemplate
                    from langchain_openai import OpenAI
                    template_new = """{question}
                    """

                    prompt = PromptTemplate.from_template(template_new)
                    llm = OpenAI(
                        temperature=0.7,
                        openai_api_key=API_KEY,
                        max_tokens=-1
                    )
                    llm_chain = prompt | llm

                    question_new = f'''
                    Bạn là một nhà phân tích dữ liệu.
                    Dựa vào kết quả "Không có giấy tờ pháp lý" có được từ mô hình, thông tin overview của bộ dữ liệu {result_dataoverview} và câu hỏi từ người dùng là {THE_PROMPT}, 
                    hãy giải thích thêm bằng tiếng Việt vì sao mô hình đã đưa ra kết quả như vậy.
                    '''

                    ketqua_cuoicung = llm_chain.invoke(question_new)
                    print('ketqua_cuoicung', ketqua_cuoicung)
                    st.write(ketqua_cuoicung)
                    
                
                # Download models
                if st.session_state["all_set"]:
                    download_col1, download_col2, download_col3 = st.columns(3)
                    with download_col1:
                        st.download_button(label="Download Model", data=st.session_state.downloadable_model1, file_name=f"{st.session_state.model1_name}.joblib", mime="application/octet-stream")
                    with download_col2:
                        st.download_button(label="Download Model", data=st.session_state.downloadable_model2, file_name=f"{st.session_state.model2_name}.joblib", mime="application/octet-stream")
                    with download_col3:
                        st.download_button(label="Download Model", data=st.session_state.downloadable_model3, file_name=f"{st.session_state.model3_name}.joblib", mime="application/octet-stream")

        # Footer
        st.divider()
        if "all_set" in st.session_state and st.session_state["all_set"]:
            if "has_been_set" not in st.session_state:
                st.session_state["has_been_set"] = True
                developer_info()
            else:
                developer_info_static()

def display_results(X_train, X_test, Y_train, Y_test):
    results = []
    st.success("Models selected based on your data!")
    
    # Data set metrics
    data_col1, data_col2, data_col3, balance_col4 = st.columns(4)
    with data_col1:
        st.metric(label="Total Data", value=len(X_train)+len(X_test), delta=None)
    with data_col2:
        st.metric(label="Training Data", value=len(X_train), delta=None)
    with data_col3:
        st.metric(label="Testing Data", value=len(X_test), delta=None)
    with balance_col4:
        st.metric(label="Balance Strategy", value=get_balance_method_name(st.session_state.balance_method), delta=None)
    
    # Model training
    model_col1, model_col2, model_col3 = st.columns(3)
    with model_col1:
        if "model1_name" not in st.session_state:
            st.session_state.model1_name = get_model_name(st.session_state.model_list[0])
        st.subheader(st.session_state.model1_name)
        with st.spinner("Model training in progress..."):
            if 'model1' not in st.session_state:
                st.session_state.model1 = train_selected_model(X_train, Y_train, st.session_state.model_list[0])
                st.session_state.downloadable_model1 = save_model(st.session_state.model1)
                
        # Model metrics
        st.write(f"The accuracy of the {st.session_state.model1_name}: ", f'\n:green[**{st.session_state.model1.score(X_test, Y_test)}**]')
        st.pyplot(confusion_metrix(st.session_state.model1_name, st.session_state.model1, X_test, Y_test))
        st.write("F1 Score: ", f':green[**{calculate_f1_score(st.session_state.model1, X_test, Y_test, st.session_state.is_binary)}**]')
        result1 = {
            "model_name":st.session_state.model1_name,
            "confusion_matrix":get_confusion_metrix(st.session_state.model1_name, st.session_state.model1, X_test, Y_test),
            "accuracy":st.session_state.model1.score(X_test, Y_test),
            "f1_score":calculate_f1_score(st.session_state.model1, X_test, Y_test, st.session_state.is_binary)
            }
        if st.session_state.model_list[0] != 2 and st.session_state['is_binary']:
            if 'fpr1' not in st.session_state:
                fpr1, tpr1 = fpr_and_tpr(st.session_state.model1, X_test, Y_test)
                st.session_state.fpr1 = fpr1
                st.session_state.tpr1 = tpr1
            st.pyplot(roc(st.session_state.model1_name, st.session_state.fpr1, st.session_state.tpr1))
            st.write(f"The AUC of the {st.session_state.model1_name}: ", f'\n:green[**{auc(st.session_state.fpr1, st.session_state.tpr1)}**]')


    with model_col2:
        if "model2_name" not in st.session_state:
            st.session_state.model2_name = get_model_name(st.session_state.model_list[1])
        st.subheader(st.session_state.model2_name)
        with st.spinner("Model training in progress..."):
            if 'model2' not in st.session_state:
                st.session_state.model2 = train_selected_model(X_train, Y_train, st.session_state.model_list[1])
                st.session_state.downloadable_model2 = save_model(st.session_state.model2)
        # Model metrics
        st.write(f"The accuracy of the {st.session_state.model2_name}: ", f'\n:green[**{st.session_state.model2.score(X_test, Y_test)}**]')
        st.pyplot(confusion_metrix(st.session_state.model2_name, st.session_state.model2, X_test, Y_test))
        st.write("F1 Score: ", f':green[**{calculate_f1_score(st.session_state.model2, X_test, Y_test, st.session_state.is_binary)}**]')
        result2 = {
            "model_name":st.session_state.model2_name,
            "confusion_matrix":get_confusion_metrix(st.session_state.model2_name, st.session_state.model2, X_test, Y_test),
            "accuracy":st.session_state.model2.score(X_test, Y_test),
            "f1_score":calculate_f1_score(st.session_state.model2, X_test, Y_test, st.session_state.is_binary)
        }
        if st.session_state.model_list[1] != 2 and st.session_state['is_binary']:
            if 'fpr2' not in st.session_state:
                fpr2, tpr2 = fpr_and_tpr(st.session_state.model2, X_test, Y_test)
                st.session_state.fpr2 = fpr2
                st.session_state.tpr2 = tpr2
            st.pyplot(roc(st.session_state.model2_name, st.session_state.fpr2, st.session_state.tpr2))
            st.write(f"The AUC of the {st.session_state.model2_name}: ", f'\n:green[**{auc(st.session_state.fpr2, st.session_state.tpr2)}**]')
        
    with model_col3:
        if "model3_name" not in st.session_state:
            st.session_state.model3_name = get_model_name(st.session_state.model_list[2])
        st.subheader(st.session_state.model3_name)
        with st.spinner("Model training in progress..."):
            if 'model3' not in st.session_state:
                st.session_state.model3 = train_selected_model(X_train, Y_train, st.session_state.model_list[2])
                st.session_state.downloadable_model3 = save_model(st.session_state.model3)
        # Model metrics
        print('Ten model t3: ', st.session_state.model3_name)
        st.write(f"The accuracy of the {st.session_state.model3_name}: ", f'\n:green[**{st.session_state.model3.score(X_test, Y_test)}**]')
        st.pyplot(confusion_metrix(st.session_state.model3_name, st.session_state.model3, X_test, Y_test))
        st.write("F1 Score: ", f':green[**{calculate_f1_score(st.session_state.model3, X_test, Y_test, st.session_state.is_binary)}**]')
        result3 = {
            "model_name":st.session_state.model3_name,
            "confusion_matrix":get_confusion_metrix(st.session_state.model3_name, st.session_state.model3, X_test, Y_test),
            "accuracy":st.session_state.model3.score(X_test, Y_test),
            "f1_score":calculate_f1_score(st.session_state.model3, X_test, Y_test, st.session_state.is_binary)
        }

        if st.session_state.model_list[2] != 2 and st.session_state['is_binary']:
            if 'fpr3' not in st.session_state:
                fpr3, tpr3 = fpr_and_tpr(st.session_state.model3, X_test, Y_test)
                st.session_state.fpr3 = fpr3
                st.session_state.tpr3 = tpr3
            st.pyplot(roc(st.session_state.model3_name, st.session_state.fpr3, st.session_state.tpr3))
            st.write(f"The AUC of the {st.session_state.model3_name}: ", f'\n:green[**{auc(st.session_state.fpr3, st.session_state.tpr3)}**]')
    
    print("Uả sao không ra kết quả dị")
    #print(st.session_state.model1)
    #scaler = StandardScaler()
    #X_new_train = scaler.fit_transform(pd.json_normalize(st.session_state.json_file))
    #print('X_new_train là: ', X_new_train)

    #new_df = clf.predict(X_new_train)
    #print('Kết quả dự đoán là: ', new_df)
    #clf = RandomForestClassifier(n_estimators=100, random_state=42)
    #clf.fit(X_train,Y_train)
    #new_df = clf.predict(X_test)
    
    results.append(result1)
    results.append(result2)
    results.append(result3)
    return results 
