import seaborn as sns
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import confusion_matrix

def distribution_histogram(df, attribute):
    plt.figure()
    sns.distplot(df[attribute])
    plt.title(f"Distribution of {attribute}")
    plt.show()

def distribution_boxplot(df, attribute):
    plt.figure()
    sns.boxenplot(data = df[attribute], palette = ["#32936f","#26a96c","#2bc016"])
    plt.title(f"Boxplot of {attribute}")
    plt.show()

def scatter_plot(df, attribute):
    plt.figure()
    sns.scatterplot(data = df, x = attribute[0], y = attribute[1])
    plt.title(f"Scatter plot of {attribute[0]} and {attribute[1]}")
    plt.show()

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
        sns.histplot(ax=axes[i], data=df, x=column, color='#25a0ff')

    # Hide additional subplots
    for ax in axes[num_plots:]: ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95) # Adjust the top to accommodate the title
    return fig

import plotly.express as px

def count_Y(df, Y_name, mapping = None):
    """
    Display the distribution of the target variable in a pie chart using matplotlib.
    Applies a mapping to the labels if provided.

    Args:
    df (DataFrame): The DataFrame containing the data.
    Y_name (str): The name of the target variable.
    mapping (dict, optional): A dictionary to map the target variable's values.
    """
    if Y_name in df.columns and df[Y_name].nunique() >= 1:
        
        mapped_data = df[Y_name].map(mapping[Y_name]) if mapping and Y_name in mapping else df[Y_name]
        value_counts = mapped_data.value_counts()
        
        plt.figure(figsize=(8, 8))
        plt.pie(value_counts, labels=value_counts.index, startangle=90, autopct='%1.1f%%')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title(f'Distribution of {Y_name}')
        plt.show()

        # value_counts = df[Y_name].value_counts()
        # fig = px.pie(names=value_counts.index, values=value_counts.values, hole=0.5)
        # fig.show()

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