import streamlit as st
from util import developer_info, developer_info_static
from src.plot import plot_clusters, correlation_matrix_plotly
from src.handle_null_value import contains_missing_value, remove_high_null, fill_null_values
from src.preprocess import convert_to_numeric, remove_duplicates, transform_data_for_clustering
from src.llm_service import decide_fill_null, decide_encode_type, decide_cluster_model
from src.pca import decide_pca, perform_PCA_for_clustering
from src.model_service import save_model, calculate_silhouette_score, calculate_calinski_harabasz_score, calculate_davies_bouldin_score, gmm_predict, estimate_optimal_clusters
from src.cluster_model import train_select_cluster_model
from src.util import contain_null_attributes_info, separate_fill_null_list, check_all_columns_numeric, non_numeric_columns_and_head, separate_decode_list, get_cluster_method_name

def start_training_model():
    st.session_state["start_training"] = True

def cluster_model_pipeline(DF, API_KEY, GPT_MODEL):
    st.divider()
    st.subheader('Data Overview')
    if 'data_origin' not in st.session_state:
        st.session_state.data_origin = DF
    st.dataframe(st.session_state.data_origin.describe(), width=1200)
    
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

    # Data Transformation
    st.subheader('Data Transformation')
    if 'data_transformed' not in st.session_state:
        st.session_state.data_transformed = transform_data_for_clustering(st.session_state.df_cleaned2)
    st.success("Data transformed by standardization and box-cox if applicable.")
    
    # PCA
    st.subheader('Principal Component Analysis')
    st.write("Deciding whether to perform PCA...")
    if 'df_pca' not in st.session_state:
        _, n_components = decide_pca(st.session_state.df_cleaned2)
        st.session_state.df_pca = perform_PCA_for_clustering(st.session_state.data_transformed, n_components)
    st.success("Completed!")

    # Splitting and Balancing
    if 'test_percentage' not in st.session_state:
        st.session_state.test_percentage = 20
    if 'balance_data' not in st.session_state:
        st.session_state.balance_data = False
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

    # Model Training
    if st.session_state['start_training']:
        with st.container():
            st.header("Modeling")
            if not st.session_state.get("data_prepared", False): 
                st.session_state.X = st.session_state.df_pca
                st.session_state.data_prepared = True

            # Decide model types:
            if "decided_model" not in st.session_state:
                st.session_state["decided_model"] = False
            if "all_set" not in st.session_state:
                st.session_state["all_set"] = False

            if not st.session_state["decided_model"]:
                with st.spinner("Deciding models based on data..."):
                    shape_info = str(st.session_state.X.shape)
                    description_info = st.session_state.X.describe().to_csv()
                    cluster_info = estimate_optimal_clusters(st.session_state.X)
                    st.session_state.default_cluster = cluster_info
                    model_dict = decide_cluster_model(shape_info, description_info, cluster_info, GPT_MODEL, API_KEY)
                    model_list = list(model_dict.values())
                    if 'model_list' not in st.session_state:
                        st.session_state.model_list = model_list
                    st.session_state.decided_model = True

            # Display results
            if st.session_state["decided_model"]:
                display_results(st.session_state.X)
                st.session_state["all_set"] = True
            
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

def display_results(X):
    st.success("Models selected based on your data!")

    # Data set metrics
    st.metric(label="Total Data", value=len(X), delta=None)
    
    # Model training
    model_col1, model_col2, model_col3 = st.columns(3)
    with model_col1:
        if "model1_name" not in st.session_state:
            st.session_state.model1_name = get_cluster_method_name(st.session_state.model_list[0])
        st.subheader(st.session_state.model1_name)

        # Slider for model parameters
        if st.session_state.model_list[0] == 2:
            st.caption('N-cluster is not applicable to DBSCAN.')
        else:
            st.caption(f'N-cluster for {st.session_state.model1_name}:')
        n_clusters1 = st.slider('N clusters', 2, 20, st.session_state.default_cluster, label_visibility="collapsed", key='n_clusters1', disabled=st.session_state.model_list[0] == 2)
        
        with st.spinner("Model training in progress..."):
            st.session_state.model1 = train_select_cluster_model(X, n_clusters1, st.session_state.model_list[0])
            st.session_state.downloadable_model1 = save_model(st.session_state.model1)
       
        if st.session_state.model_list[0] != 3:
            label1 = st.session_state.model1.labels_
        else:
            label1 = gmm_predict(X, st.session_state.model1)

        # Visualization
        st.pyplot(plot_clusters(X, label1))
        # Model metrics
        st.write(f"Silhouette score: ", f'\n:green[**{calculate_silhouette_score(X, label1)}**]')
        st.write(f"Calinski-Harabasz score: ", f'\n:green[**{calculate_calinski_harabasz_score(X, label1)}**]')
        st.write(f"Davies-Bouldin score: ", f'\n:green[**{calculate_davies_bouldin_score(X, label1)}**]')

    with model_col2:
        if "model2_name" not in st.session_state:
            st.session_state.model2_name = get_cluster_method_name(st.session_state.model_list[1])
        st.subheader(st.session_state.model2_name)

        # Slider for model parameters
        if st.session_state.model_list[1] == 2:
            st.caption('N-cluster is not applicable to DBSCAN.')
        else:
            st.caption(f'N-cluster for {st.session_state.model2_name}:')
        n_clusters2 = st.slider('N clusters', 2, 20, st.session_state.default_cluster, label_visibility="collapsed", key='n_clusters2', disabled=st.session_state.model_list[1] == 2)

        with st.spinner("Model training in progress..."):
            st.session_state.model2 = train_select_cluster_model(X, n_clusters2, st.session_state.model_list[1])
            st.session_state.downloadable_model2 = save_model(st.session_state.model2)

        if st.session_state.model_list[1] != 3:
            label2 = st.session_state.model2.labels_
        else:
            label2 = gmm_predict(X, st.session_state.model2)

        # Visualization
        st.pyplot(plot_clusters(X, label2))
        # Model metrics
        st.write(f"Silhouette score: ", f'\n:green[**{calculate_silhouette_score(X, label2)}**]')
        st.write(f"Calinski-Harabasz score: ", f'\n:green[**{calculate_calinski_harabasz_score(X, label2)}**]')
        st.write(f"Davies-Bouldin score: ", f'\n:green[**{calculate_davies_bouldin_score(X, label2)}**]')
        
    with model_col3:
        if "model3_name" not in st.session_state:
            st.session_state.model3_name = get_cluster_method_name(st.session_state.model_list[2])
        st.subheader(st.session_state.model3_name)

        # Slider for model parameters
        if st.session_state.model_list[2] == 2:
            st.caption('N-cluster is not applicable to DBSCAN.')
        else:
            st.caption(f'N-cluster for {st.session_state.model3_name}:')
        n_clusters3 = st.slider('N clusters', 2, 20, st.session_state.default_cluster, label_visibility="collapsed", key='n_clusters3', disabled=st.session_state.model_list[2] == 2)

        with st.spinner("Model training in progress..."):
            st.session_state.model3 = train_select_cluster_model(X, n_clusters3, st.session_state.model_list[2])
            st.session_state.downloadable_model3 = save_model(st.session_state.model3)

        if st.session_state.model_list[2] != 3:
            label3 = st.session_state.model3.labels_
        else:
            label3 = gmm_predict(X, st.session_state.model3)

        # Visualization
        st.pyplot(plot_clusters(X, label3))
        # Model metrics
        st.write(f"Silhouette score: ", f'\n:green[**{calculate_silhouette_score(X, label3)}**]')
        st.write(f"Calinski-Harabasz score: ", f'\n:green[**{calculate_calinski_harabasz_score(X, label3)}**]')
        st.write(f"Davies-Bouldin score: ", f'\n:green[**{calculate_davies_bouldin_score(X, label3)}**]')
