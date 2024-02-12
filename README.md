# Streamline Analyst: Data Analysis AI Agent

Streamline Analyst 🪄 is an open source, Large Language Models (LLMs) driven **Data Analysis Agent**. In our application, common tasks in data analysis such as data cleaning and preprocessing will be automatically handled by AI. It can also automatically identify target objects, divide test sets, select the most appropriate model for modeling based on data, visualize results evaluation, etc. What user needs to do is to **select the data file**, **choose an analysis mode**, and **click start**. 

Streamline Analyst is aimed to accelerate and simplify the entire process of data analysis. Regardless of whether the user has professional data analysis skills, we hope to help users process data and visualization with the highest efficiency🚀, and complete high-performance modeling tasks with the optimal strategies🔮.

***Natural Language Processing (NLP)***, ***neural network***, ***object detection (using YOLO)*** may be added in subsequent versions.

Features of current version
---------------------------
* LLM determines the **target variable**
* LLM decision strategy for **null value handling**:
    * Mean filling, median filling, mode filling, interpolation filling, introduction of new categories, etc.
* LLM decision **data encoding**: 
    * One-hot encoding, integer mapping, label encoding
* Principal Component Analysis (**PCA**) dimensionality reduction
* Handling duplicate entities
* **Box-Cox transformation** and **normalization**
* LLM decision strategy to **balance** the number of entities of the target variable: 
    * Random over-sampling, SMOTE, ADASYN
* LLM determines the proportion of the data set (can also be adjusted manually)
* LLM determines the **suitable models** based on the data and **starts training**
* Recommend the **number of clusters** through the Elbow Rule and Silhouette Coefficient, and real-time adjustment of the cluster number
* All processed data and models can be downloaded

* Modeling tasks supported:

| **Classification Models**      | **Clustering Models**       | **Regression Models**             |
|--------------------------------|-----------------------------|-----------------------------------|
| Logistic regression            | K-means clustering          | Linear regression                 |
| Random forest                  | DBSCAN                      | Ridge regression                  |
| Support vector machine         | Gaussian mixture model      | Lasso regression                  |
| Gradient boosting machine      | Hierarchical clustering     | Elastic net regression            |
| Gaussian Naive Bayes           | Spectral clustering         | Random forest regression          |
| AdaBoost                       | etc.                        | Gradient boosting regression      |
| XGBoost                        |                             | etc.                              |

* Real-time calculation of model indicators and result visualization:

| **Classification Metrics & Plots** | **Clustering Metrics & Plots** | **Regression Metrics & Plots**        |
|------------------------------------|--------------------------------|---------------------------------------|
| Model score                        | Silhouette score               | R-squared score                       |
| Confusion matrix                   | Calinski-Harabasz score        | Mean square error (MSE)               |
| AUC                                | Davies-Bouldin score           | Root mean square error (RMSE)         |
| F1 score                           | Cluster scatter plot           | Absolute error (MAE)                  |
| ROC plot                           | etc.                           | Residual plot                         |
| etc.                               |                                | Predicted value vs actual value plot  |
|                                    |                                | Quantile-Quantile plot                |

* Visual analysis(No API Key needed):
    * Single attribute data visualization
    * Multi-attribute data visualization
    * Three-dimensional plot
    * Word Cloud frequency analysis
    * World heat map
    * etc.

*Note: The uploaded data and API Key are for one-time use and WILL NOT be saved or shared in any form.*

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
