# Streamline-Analyst

`Streamline Analyst` is an open source, Large Language Models (LLMs) driven **Data Analysis Agent**. Common tasks in data analysis such as data cleaning and preprocessing will be automatically handled by AI. It can also automatically identify target objects, divide test sets, select the most appropriate model for modeling based on data, visualize results evaluation, etc. What user needs to do is to select the data file, choose an analysis mode, and click start. 

`Streamline Analyst` is aimed to accelerate and simplify the entire process of data analysis. Regardless of whether the user has professional data analysis skills, we hope to help users process data and visualization with the highest efficiency, and complete high-performance modeling tasks with the optimal strategies.

*Natural Language Processing (NLP)*, *neural network*, *object detection (using YOLO)* may be added in subsequent versions.

Features of current version
---------------------------
* LLM determines the target variable
* LLM decision strategy for null value handling:
    * Mean filling
    * Median filling
    * Mode filling
    * Interpolation filling
    * Introduction of new categories
* LLM decision data encoding: 
    * Integer mapping
    * One-hot encoding
    * Label encoding
* Principal Component Analysis (PCA) dimensionality reduction
* Handling duplicate entities
* Box-Cox transformation and normalization
* LLM decision strategy to balance the number of entities of the target variable: 
    * Random over-sampling
    * SMOTE
    * ADASYN
* LLM determines the proportion of the data set (can also be adjusted manually)
* LLM determines the suitable models based on the data and starts training
* Recommend the number of clusters through the Elbow Rule and Silhouette Coefficient, and real-time adjustment of the cluster number
* All processed data and models can be downloaded

* Modeling tasks supported:
    * Logistic regression
    * Random forest
    * Support vector machine
    * Gradient boosting machine
    * Gaussian Naive Bayes
    * AdaBoost
    * XGBoost
    * K-means clustering
    * DBSCAN
    * Gaussian mixture model
    * Hierarchical clustering
    * Spectral clustering
    * Linear regression
    * Ridge regression
    * Lasso regression
    * Elastic net regression
    * Random forest regression
    * Gradient boosting regression

* Real-time calculation of model indicators and result visualization:
    * Model score
    * Confusion matrix
    * AUC
    * F1 score
    * ROC plot
    * Silhouette score
    * Calinski-Harabasz score
    * Davies-Bouldin score
    * Cluster scatter plot
    * R-squared score
    * Mean square error (MSE)
    * Root mean square error (RMSE)
    * Absolute error (MAE)
    * Residual plot
    * Predicted value vs actual value plot
    * Quantile-Quantile plot
    * etc.

* Visual analysis(No API Key needed):
    * Single attribute data visualization
    * Multi-attribute data visualization
    * Three-dimensional plot
    * Word Cloud frequency analysis
    * World heat map
    * etc.

Note: The uploaded data and API Key are for one-time use and WILL NOT be saved or shared in any form.

Demo
----


Getting started
---------------

### Prerequisites

To run `app.py`, you'll need:
* [Python 3.11.5](https://www.python.org/downloads/)
* [OpenAI API Key](https://openai.com/blog/openai-api)
    * OpenAI: Note that the free quota does not support GPT-4
    
### Installation
1. Install the required packages

```
pip install -r requirements.txt
```

2. Run `app.py` on your local machine

```
streamlit run app.py
```
