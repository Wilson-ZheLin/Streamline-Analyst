import streamlit as st
from util import developer_info
from src.plot import list_all, correlation_matrix, plot_clusters, correlation_matrix_plotly
from src.handle_null_value import contains_missing_value, remove_high_null, fill_null_values
from src.preprocess import convert_to_numeric, remove_duplicates
from src.llm_service import decide_fill_null, decide_encode_type
from src.pca import decide_pca, perform_pca
from src.model_service import save_model, standardize_data, calculate_silhouette_score, calculate_calinski_harabasz_score, calculate_davies_bouldin_score, gmm_predict
from src.cluster_model import KMeans_train, DBSCAN_train, GaussianMixture_train
from src.util import contain_null_attributes_info, separate_fill_null_list, check_all_columns_numeric, non_numeric_columns_and_head, separate_decode_list

def start_training_model():
    st.session_state["start_training"] = True

def cluster_model_pipeline(DF, API_KEY, GPT_MODEL):
    st.divider()
    st.subheader('Data Overview')
    if 'data_origin' not in st.session_state:
        st.session_state.data_origin = DF
    st.dataframe(st.session_state.data_origin.describe(), width=1200)
    # st.pyplot(list_all(st.session_state.data_origin))
    
    # Data Imputation
    st.subheader('Handle and Impute Missing Values')
    if "contain_null" not in st.session_state:
            st.session_state.contain_null = contains_missing_value(st.session_state.data_origin)

    if 'filled_df' not in st.session_state:
        if st.session_state.contain_null:
            with st.status("Processing **missing values** in the data...", expanded=True) as status:
                st.write("Filtering out high-frequency missing rows and columns...")
                filled_df = remove_high_null(DF)
                st.write("Large language model analysis...")
                attributes, types_info, description_info = contain_null_attributes_info(filled_df)
                fill_result_dict = decide_fill_null(attributes, types_info, description_info, GPT_MODEL, API_KEY)
                st.write("Imputing missing values...")
                mean_list, median_list, mode_list, new_category_list, interpolation_list = separate_fill_null_list(fill_result_dict)
                filled_df = fill_null_values(filled_df, mean_list, median_list, mode_list, new_category_list, interpolation_list)
                # Store the imputed DataFrame in session_state
                st.session_state.filled_df = filled_df
                DF = filled_df
                status.update(label='Missing value processing completed!', state="complete", expanded=False)
            st.download_button(
                label="Download Data with Missing Values Imputed",
                data=st.session_state.filled_df.to_csv(index=False).encode('utf-8'),
                file_name="imputed_missing_values.csv",
                mime='text/csv')
        else:
            st.session_state.filled_df = DF
            st.success("No missing values detected. Processing skipped.")
    else:
        st.success("Missing value processing completed!")
        if st.session_state.contain_null:
            st.download_button(
                label="Download Data with Missing Values Imputed",
                data=st.session_state.filled_df.to_csv(index=False).encode('utf-8'),
                file_name="imputed_missing_values.csv",
                mime='text/csv')

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
                convert_int_cols, one_hot_cols, drop_cols = separate_decode_list(encode_result_dict, "")
                encoded_df, mappings = convert_to_numeric(DF, convert_int_cols, one_hot_cols, drop_cols)
                # Store the imputed DataFrame in session_state
                st.session_state.encoded_df = encoded_df
                DF = encoded_df
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
    st.subheader('Correlation Between Attributes')
    st.plotly_chart(correlation_matrix_plotly(st.session_state.df_cleaned1))

    # Remove duplicate entities
    st.subheader('Remove Duplicate Entities')
    if 'df_cleaned2' not in st.session_state:
        st.session_state.df_cleaned2 = remove_duplicates(st.session_state.df_cleaned1)
        # DF = remove_duplicates(DF)
    st.info("Duplicate rows removed.")
    
    # PCA
    st.subheader('Principal Component Analysis')
    st.write("Deciding whether to perform PCA...")
    if 'df_pca' not in st.session_state:
        to_perform_pca, n_components = decide_pca(st.session_state.df_cleaned2)
        if 'to_perform_pca' not in st.session_state:
            st.session_state.to_perform_pca = to_perform_pca
        if st.session_state.to_perform_pca:
            st.session_state.df_pca = perform_pca(st.session_state.df_cleaned2, n_components, "")
        else:
            st.session_state.df_pca = st.session_state.df_cleaned2
    st.success("Completed!")

    # Splitting and Balancing
    if 'test_percentage' not in st.session_state:
        st.session_state.test_percentage = 20
    if 'balance_data' not in st.session_state:
        st.session_state.balance_data = False

    # Model Training
    if "start_training" not in st.session_state:
        st.session_state["start_training"] = False
    if 'model_trained' not in st.session_state:
        st.session_state['model_trained'] = False

    splitting_column, balance_column = st.columns(2)
    with splitting_column:
        st.subheader(':grey[Data Splitting]')
        st.caption('Data splitting is not applicable to clustering models.')
        st.slider('Percentage of test set', 1, 25, st.session_state.test_percentage, key='test_percentage', disabled=True)
    
    with balance_column:
        st.metric(label="Test Data", value="--%", delta=None)
        st.toggle('Class Balancing', value=st.session_state.balance_data, key='to_perform_balance', disabled=True)
        st.caption('Class balancing is not applicable to clustering models.')
    
    st.button("Start Training Model", on_click=start_training_model, type="primary", disabled=st.session_state['start_training'])

    if st.session_state['start_training']:
        with st.container():
            st.header("Modeling")
            X = st.session_state.df_pca

            # Standardize the data
            if "data_prepared" not in st.session_state:
                st.session_state.data_prepared = False
            if not st.session_state.data_prepared:
                if not st.session_state.to_perform_pca:
                    st.session_state.X = standardize_data(X)
                else:
                    st.session_state.X = X
                st.session_state.data_prepared = True

            # Start training model
            if "all_set" not in st.session_state:
                st.session_state["all_set"] = False

            display_results(st.session_state.X)
            
            if not st.session_state["all_set"]:
                st.session_state["all_set"] = True
            
            if st.session_state["all_set"]:
                download_col1, download_col2, download_col3 = st.columns(3)
                with download_col1:
                    st.download_button(label="Download Model", data=st.session_state.downloadable_model1, file_name=f"{st.session_state.model1_name}.joblib", mime="application/octet-stream")
                with download_col2:
                    st.download_button(label="Download Model", data=st.session_state.downloadable_model2, file_name=f"{st.session_state.model2_name}.joblib", mime="application/octet-stream")
                with download_col3:
                    st.download_button(label="Download Model", data=st.session_state.downloadable_model3, file_name=f"{st.session_state.model3_name}.joblib", mime="application/octet-stream")

    st.divider()
    if "all_set" in st.session_state and st.session_state["all_set"]: developer_info()

def display_results(X):

    # Data set metrics
    st.metric(label="Total Data", value=len(X), delta=None)
    
    # Model training
    model_col1, model_col2, model_col3 = st.columns(3)
    with model_col1:
        if "model1_name" not in st.session_state:
            st.session_state.model1_name = "K-Means Clustering"
        st.subheader(st.session_state.model1_name)
        # Slider for model parameters
        st.caption('N-cluster for K-Means:')
        n_clusters1 = st.slider('N clusters', 2, 20, 3, label_visibility="collapsed", key='n_clusters1')
        
        with st.spinner("Model training in progress..."):
            st.session_state.model1 = KMeans_train(X, n_clusters=n_clusters1)
            st.session_state.downloadable_model1 = save_model(st.session_state.model1)
       
        # Visualization
        st.pyplot(plot_clusters(X, st.session_state.model1.labels_))
        # Model metrics
        st.write(f"Silhouette score: ", f'\n:green[**{calculate_silhouette_score(X, st.session_state.model1.labels_)}**]')
        st.write(f"Calinski-Harabasz score: ", f'\n:green[**{calculate_calinski_harabasz_score(X, st.session_state.model1.labels_)}**]')
        st.write(f"Davies-Bouldin score: ", f'\n:green[**{calculate_davies_bouldin_score(X, st.session_state.model1.labels_)}**]')

    with model_col2:
        if "model2_name" not in st.session_state:
            st.session_state.model2_name = "DBSCAN"
        st.subheader(st.session_state.model2_name)
        # Slider for model parameters
        st.caption('N-cluster is not applicable to DBSCAN.')
        n_clusters2 = st.slider('N clusters', 2, 20, 3, label_visibility="collapsed", disabled=True, key='n_clusters2')
        
        with st.spinner("Model training in progress..."):
            st.session_state.model2 = DBSCAN_train(X)
            st.session_state.downloadable_model2 = save_model(st.session_state.model2)
       
        # Visualization
        st.pyplot(plot_clusters(X, st.session_state.model2.labels_))
        # Model metrics
        st.write(f"Silhouette score: ", f'\n:green[**{calculate_silhouette_score(X, st.session_state.model2.labels_)}**]')
        st.write(f"Calinski-Harabasz score: ", f'\n:green[**{calculate_calinski_harabasz_score(X, st.session_state.model2.labels_)}**]')
        st.write(f"Davies-Bouldin score: ", f'\n:green[**{calculate_davies_bouldin_score(X, st.session_state.model2.labels_)}**]')

    with model_col3:
        if "model3_name" not in st.session_state:
            st.session_state.model3_name = "Gaussian Mixture"
        st.subheader(st.session_state.model3_name)
        # Slider for model parameters
        st.caption('N-Component for Gaussian Mixture:')
        n_clusters3 = st.slider('N components', 2, 20, 3, label_visibility="collapsed", key='n_clusters3')
        
        with st.spinner("Model training in progress..."):
            st.session_state.model3 = GaussianMixture_train(X, n_components=n_clusters3)
            st.session_state.downloadable_model3 = save_model(st.session_state.model3)
       
        # Visualization
        st.pyplot(plot_clusters(X, gmm_predict(X, st.session_state.model3)))
        # Model metrics
        st.write(f"Silhouette score: ", f'\n:green[**{calculate_silhouette_score(X, gmm_predict(X, st.session_state.model3))}**]')
        st.write(f"Calinski-Harabasz score: ", f'\n:green[**{calculate_calinski_harabasz_score(X, gmm_predict(X, st.session_state.model3))}**]')
        st.write(f"Davies-Bouldin score: ", f'\n:green[**{calculate_davies_bouldin_score(X, gmm_predict(X, st.session_state.model3))}**]')
