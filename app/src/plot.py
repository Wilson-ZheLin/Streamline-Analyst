import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

def distribution_histogram(df, attribute):
    if df[attribute].dtype == 'object' or pd.api.types.is_categorical_dtype(df[attribute]):
        codes, uniques = pd.factorize(df[attribute])
        temp_df = pd.DataFrame({attribute: codes})
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(temp_df[attribute], ax=ax, discrete=True)
        ax.set_xticks(range(len(uniques)))
        ax.set_xticklabels(uniques, rotation=45, ha='right')
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[attribute], ax=ax)

    ax.set_title(f"Distribution of {attribute}")
    return fig

def distribution_boxplot(df, attribute):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxenplot(data=df.attribute, palette=["#32936f", "#26a96c", "#2bc016"])
    ax.set_title(f"Boxplot of {attribute}")
    return fig

@st.cache_data
def correlation_matrix(df):
    plt.figure(figsize=(16, 12))
    sns.set(font_scale=0.9)
    sns.heatmap(df.corr(), annot=True, cmap='viridis', annot_kws={"size": 12})
    return plt.gcf()

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

import plotly.express as px

def count_Y(df, Y_name):
    if Y_name in df.columns and df[Y_name].nunique() >= 1:
        value_counts = df[Y_name].value_counts()
        fig = px.pie(names=value_counts.index, values=value_counts.values, title=f'Distribution of {Y_name}', hole=0.5, color_discrete_sequence=px.colors.sequential.Cividis_r)
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
    fig, ax = plt.subplots()
    for label in set(labels):
        idx = labels == label
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f'Cluster {label}', s=50)
    
    ax.set_title('Cluster Scatter Plot')
    ax.legend()
    return fig