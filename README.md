# Streamline Analyst: A Data Analysis AI Agent

Languages / 语言选择: [English](https://github.com/Wilson-ZheLin/Streamline-Analyst) | [中文](https://github.com/Wilson-ZheLin/Streamline-Analyst/blob/main/README_CN.md)

Streamline Analyst 🪄 is a cutting-edge, open-source application powered by Large Language Models (LLMs) designed to revolutionize data analysis. This **Data Analysis Agent** effortlessly automates all the tasks such as data cleaning, preprocessing, and even complex operations like identifying target objects, partitioning test sets, and selecting the best-fit models based on your data. With Streamline Analyst, results visualization and evaluation become seamless.

Here's how it simplifies your workflow: just **select your data file**, **pick an analysis mode**, and **hit start**. Streamline Analyst aims to expedite the data analysis process, making it accessible to all, regardless of their expertise in data analysis. It's built to empower users to process data and achieve high-quality visualizations with unparalleled efficiency🚀, and to execute high-performance modeling with the best strategies🔮.

**Try Our Live Demo Here**: [Streamline Analyst](https://streamline.streamlit.app)

<small>When utilizing `GPT-4 turbo`, the cost for each comprehensive end-to-end API request is roughly <strong>$0.02</strong>.</small>

Your data's privacy and security are paramount; rest assured, uploaded data and API Keys are strictly for one-time use and are neither saved nor shared.

![Screenshot 2024-02-12 at 16 01 01](https://github.com/Wilson-ZheLin/Streamline-Analyst/assets/145169519/4167b04c-0853-4703-87a4-6c2994e30f9e)

Looking ahead, we plan to enhance Streamline Analyst with advanced features like ***Natural Language Processing (NLP)***, ***neural networks***, and ***object detection (utilizing YOLO)***, broadening its capabilities to meet more diverse data analysis needs.

Demo
----

https://github.com/Wilson-ZheLin/Streamline-Analyst/assets/145169519/1d30faca-f474-42fd-b20b-c93ed7cf6d13

**Demo link available at**: [Streamline Analyst](https://streamline.streamlit.app)

Current Version Features
------------------------
* **Target Variable Identification**: LLMs adeptly pinpoint the target variable
* **Null Value Management**: Choose from a variety of strategies such as mean, median, mode filling, interpolation, or introducing new categories for handling missing data, all recommended by LLMs
* **Data Encoding Tactics**: Automated suggestions and completions for the best encoding methods, including one-hot, integer mapping, and label encoding
* **Dimensionality Reduction with PCA**
* **Duplicate Entity Resolution**
* **Data Transformation and Normalization**: Utilize Box-Cox transformation and normalization techniques to improve data distribution and scalability
* **Balancing Target Variable Entities**: LLM-recommended methods like random over-sampling, SMOTE, and ADASYN help balance data sets, crucial for unbiased model training
* **Data Set Proportion Adjustment**: LLM determines the proportion of the data set (can also be adjusted manually)
* **Model Selection and Training**: Based on your data, LLMs recommend and initiate training with the most suitable models
* **Cluster Number Recommendation**: Leveraging the Elbow Rule and Silhouette Coefficient for optimal cluster numbers, with the flexibility of real-time adjustments

All processed data and models are made available for download, offering a comprehensive, user-friendly data analysis toolkit.

### Modeling and Results Visualization:

![Screenshot 2024-02-12 at 16 10 35](https://github.com/Wilson-ZheLin/Streamline-Analyst/assets/145169519/423da7be-63f1-491d-9ebe-6a788c440c40)

### Automated Workflow Interface:

![Screenshot 2024-02-12 at 16 20 19](https://github.com/Wilson-ZheLin/Streamline-Analyst/assets/145169519/9d04d5f2-4f2a-44eb-ab8b-c07c8c0c5a53)

### Supported Modeling tasks:

| **Classification Models**        | **Clustering Models**         | **Regression Models**               |
|----------------------------------|-------------------------------|-------------------------------------|
| Logistic regression              | K-means clustering            | Linear regression                   |
| Random forest                    | DBSCAN                        | Ridge regression                    |
| Support vector machine           | Gaussian mixture model        | Lasso regression                    |
| Gradient boosting machine        | Hierarchical clustering       | Elastic net regression              |
| Gaussian Naive Bayes             | Spectral clustering           | Random forest regression            |
| AdaBoost                         | etc.                          | Gradient boosting regression        |
| XGBoost                          |                               | etc.                                |

### Real-time calculation of model indicators and result visualization:

| **Classification Metrics & Plots** | **Clustering Metrics & Plots** | **Regression Metrics & Plots**        |
|------------------------------------|--------------------------------|---------------------------------------|
| Model score                        | Silhouette score               | R-squared score                       |
| Confusion matrix                   | Calinski-Harabasz score        | Mean square error (MSE)               |
| AUC                                | Davies-Bouldin score           | Root mean square error (RMSE)         |
| F1 score                           | Cluster scatter plot           | Absolute error (MAE)                  |
| ROC plot                           | etc.                           | Residual plot                         |
| etc.                               |                                | Predicted value vs actual value plot  |
|                                    |                                | Quantile-Quantile plot                |

### Visual Analysis Toolkit:

Streamline Analyst 🪄 offers an array of intuitive visual tools for enhanced data insight, **without the need for an API Key**:

* **Single Attribute Visualization**: Insightful views into individual data aspects
* **Multi-Attribute Visualization**: Comprehensive analysis of variable interrelations
* **Three-Dimensional Plotting**: Advanced 3D representations for complex data relationships
* **Word Clouds**: Key themes and concepts highlighted through word frequency
* **World Heat Maps**: Geographic trends and distributions made visually accessible

Local Installation
------------------

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
