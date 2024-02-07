import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
import nltk
from nltk import regexp_tokenize
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix

# Single attribute visualization
def distribution_histogram(df, attribute):
    if df[attribute].dtype == 'object' or pd.api.types.is_categorical_dtype(df[attribute]):
        codes, uniques = pd.factorize(df[attribute])
        temp_df = pd.DataFrame({attribute: codes})
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(temp_df[attribute], ax=ax, discrete=True, color='#e17160')
        ax.set_xticks(range(len(uniques)))
        ax.set_xticklabels(uniques, rotation=45, ha='right')
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[attribute], ax=ax, color='#e17160')

    ax.set_title(f"Distribution of {attribute}")
    return fig

def distribution_boxplot(df, attribute):
    if df[attribute].dtype == 'object' or pd.api.types.is_categorical_dtype(df[attribute]):
        return -1
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxenplot(data=df[attribute], palette=["#32936f", "#26a96c", "#2bc016"])
    ax.set_title(f"Boxplot of {attribute}")
    return fig

def count_Y(df, Y_name):
    if Y_name in df.columns and df[Y_name].nunique() >= 1:
        value_counts = df[Y_name].value_counts()
        fig = px.pie(names=value_counts.index, 
                     values=value_counts.values, 
                     title=f'Distribution of {Y_name}', 
                     hole=0.5, 
                     color_discrete_sequence=px.colors.sequential.Cividis_r)
        return fig

def density_plot(df, column_name):
    if column_name in df.columns:
        fig = px.density_contour(df, x=column_name, y=column_name,
                                 title=f'Density Plot of {column_name}',
                                 color_discrete_sequence=px.colors.sequential.Inferno)
        return fig

# Mutiple attribute visualization
def box_plot(df, column_names):
    if len(column_names) > 1 and not all(df[column_names].dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        return -1
    valid_columns = [col for col in column_names if col in df.columns]
    if valid_columns:
        fig = px.box(df, y=valid_columns,
                     title=f'Box Plot of {", ".join(valid_columns)}',
                     color_discrete_sequence=px.colors.sequential.Cividis_r)
        return fig

def violin_plot(df, column_names):
    if len(column_names) > 1 and not all(df[column_names].dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        return -1
    valid_columns = [col for col in column_names if col in df.columns]
    if valid_columns:
        fig = px.violin(df, y=valid_columns,
                        title=f'Violin Plot of {", ".join(valid_columns)}',
                        color_discrete_sequence=px.colors.sequential.Cividis_r)
        return fig

def strip_plot(df, column_names):
    if len(column_names) > 1 and not all(df[column_names].dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        return -1
    valid_columns = [col for col in column_names if col in df.columns]
    if valid_columns:
        fig = px.strip(df, y=valid_columns,
                       title=f'Strip Plot of {", ".join(valid_columns)}',
                       color_discrete_sequence=px.colors.sequential.Cividis_r)
        return fig

def multi_plot_scatter(df, selected_attributes):
    if len(selected_attributes) < 2:
        return -1
    
    plt.figure(figsize=(10, 6))
    if df[selected_attributes[0]].dtype not in [np.float64, np.int64]:
        x, x_labels = pd.factorize(df[selected_attributes[0]])
        plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, rotation=45)
    else:
        x = df[selected_attributes[0]]
    
    if df[selected_attributes[1]].dtype not in [np.float64, np.int64]:
        y, y_labels = pd.factorize(df[selected_attributes[1]])
        plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels)
    else:
        y = df[selected_attributes[1]]
    
    plt.scatter(x, y, c=np.linspace(0, 1, len(df)), cmap='viridis')
    plt.colorbar()
    plt.xlabel(selected_attributes[0])
    plt.ylabel(selected_attributes[1])
    plt.title(f'Scatter Plot of {selected_attributes[0]} vs {selected_attributes[1]}')
    return plt.gcf()
    
def multi_plot_line(df, selected_attributes):
    if not all(df[selected_attributes].dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        return -1
    if len(selected_attributes) >= 2:
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(selected_attributes)))
        for i, attribute in enumerate(selected_attributes):
            plt.plot(df.index, df[attribute], marker='', linewidth=2, color=colors[i], label=attribute)
        plt.legend()
        plt.xlabel(selected_attributes[0])
        plt.ylabel(selected_attributes[1])
        plt.title(f'Line Plot of {selected_attributes[0]} vs {selected_attributes[1]}')
        return plt.gcf()
    else:
        return -2
    
def multi_plot_heatmap(df, selected_attributes):
    if not all(df[selected_attributes].dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        return -1
    if len(selected_attributes) >= 1:
        sns.set_theme()
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[selected_attributes].corr(), annot=True, cmap='viridis')
        plt.title('Heatmap of Correlation')
        return plt.gcf()

# Overall visualization
@st.cache_data
def correlation_matrix(df):
    plt.figure(figsize=(16, 12))
    sns.set(font_scale=0.9)
    sns.heatmap(df.corr(), annot=True, cmap='viridis', annot_kws={"size": 12})
    return plt.gcf()

@st.cache_data
def correlation_matrix_plotly(df):
    corr_matrix = df.corr()
    labels = corr_matrix.columns
    text = [[f'{corr_matrix.iloc[i, j]:.2f}' for j in range(len(labels))] for i in range(len(labels))]
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=labels,
        y=labels,
        colorscale='Viridis',
        colorbar=dict(title='Correlation'),
        text=text,
        hoverinfo='text',
    ))
    fig.update_layout(
        title='Correlation Matrix Between Attributes',
        xaxis=dict(tickmode='linear'),
        yaxis=dict(tickmode='linear'),
        width=800,
        height=700,
    )
    fig.update_layout(font=dict(size=10))
    return fig

@st.cache_data
def list_all(df, max_plots=16):

    # Calculate the number of plots to display (up to 16)
    num_plots = min(len(df.columns), max_plots)
    nrows = int(np.ceil(num_plots / 4))
    ncols = min(num_plots, 4)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    fig.suptitle('Attribute Distributions', fontsize=20)
    plt.style.use('ggplot')
    sns.set(style="darkgrid")

    # if only one plot, convert to list
    if num_plots == 1: axes = [axes]

    # Flatten the axes array
    axes = axes.flatten()

    # Display the histograms
    for i, column in enumerate(df.columns[:num_plots]):
        sns.histplot(ax=axes[i], data=df, x=column, color='#1867ac')

    # Hide additional subplots
    for ax in axes[num_plots:]: ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95) # Adjust the top to accommodate the title
    return fig

def confusion_metrix(model_name, model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    matrix = confusion_matrix(Y_test, Y_pred)
    plt.figure(figsize=(10, 7)) # temporary
    sns_heatmap = sns.heatmap(matrix, annot=True, cmap='Blues', fmt='g', annot_kws={"size": 20})
    plt.title(f"Confusion Matrix for {model_name}", fontsize=20)
    plt.xlabel('Predicted labels', fontsize=16)
    plt.ylabel('True labels', fontsize=16)
    return sns_heatmap.figure

def roc(model_name, fpr, tpr):
    fig = plt.figure()
    plt.style.use('ggplot')
    plt.plot([0,1],[0,1],'k--')
    plt.plot(fpr, tpr, label=model_name)
    plt.xlabel('False Positive rate')
    plt.ylabel('True Positive rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='best')
    plt.xticks(rotation=45)
    return fig

def plot_clusters(X, labels):
    sns.set(style="whitegrid")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    unique_labels = set(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    fig, ax = plt.subplots()
    for color, label in zip(colors, unique_labels):
        idx = labels == label
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1], color=color, label=f'Cluster {label}', s=50)
    
    ax.set_title('Cluster Scatter Plot')
    ax.legend()
    return fig

# Advanced Visualization
@st.cache_data
def word_cloud_plot(text):
    try:
        words = regexp_tokenize(text, pattern='\w+')
        text_dist = nltk.FreqDist([w for w in words])
        wordcloud = WordCloud(width=1200, height=600, background_color ='white').generate_from_frequencies(text_dist)
        fig, ax = plt.subplots(figsize=(10, 7.5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig
    except:
        return -1

@st.cache_data
def world_map(df, country_column, key_attribute):
    try:
        hover_data_columns = [col for col in df.columns if col != country_column]
        fig = px.choropleth(df, locations="iso_alpha",
                            color=key_attribute, 
                            hover_name=country_column,
                            hover_data=hover_data_columns,
                            color_continuous_scale=px.colors.sequential.Cividis,
                            projection="equirectangular",)
        return fig
    except:
        return -1

@st.cache_data
def scatter_3d(df, x, y, z):
    try:
        return px.scatter_3d(df, x=x, y=y, z=z, color=z, color_continuous_scale=px.colors.sequential.Viridis)
    except:
        return -1
