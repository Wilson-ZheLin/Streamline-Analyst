# Streamline Analyst: 数据分析领域的AI Agent

Languages / 语言选择: [中文](https://github.com/Wilson-ZheLin/Streamline-Analyst/blob/main/README_CN.md) | [English](https://github.com/Wilson-ZheLin/Streamline-Analyst)

**Streamline Analyst**🪄是一个开源的基于GPT-4这样的大语言模型的应用，目标简化数据分析中从数据清洗到模型测试的全部流程。分类预测、聚类、回归、数据集可视化、数据预处理、编码、特征选择、目标属性判断、可视化、最佳模型选择等等任务都可自主决策和执行。用户需要做的只有**选择数据文件**、**选择分析模式**，剩下的工作就可以让AI来接管了🔮。所有处理后的数据和训练的模型都可下载。

Token花费：<small>以使用`GPT-4 turbo`模型为例，每次完整的全流程分析花费的token大约为<strong>$0.02</strong>。</small>

*所有上传的数据和API Keys不会以任何形式储存或分享！

![Screenshot 2024-02-12 at 16 01 01](https://github.com/Wilson-ZheLin/Streamline-Analyst/assets/145169519/4167b04c-0853-4703-87a4-6c2994e30f9e)

未来版本预期更新功能：***自然语言处理 (NLP)***、***卷积/循环神经网络***、***目标检测 (基于YOLO)***...

主页
----

https://github.com/Wilson-ZheLin/Streamline-Analyst/assets/145169519/56884ea5-1426-4126-b210-e7e529a34c4a

**在线Demo链接**: [Streamline Analyst](https://streamline.streamlit.app)

当前版本功能
----------

* **目标变量识别**: 若LLM无法确定，则提醒用户选择
* **空值管理**: 由LLM根据每列数据信息从均值、中位数、众数填充、插值，或引入新类别等策略中选择
* **数据编码**: 根据每列数据信息判断使用：独热编码、整数映射或标签编码
* **PCA降维**
* **处理重复实体**
* **数据转换和标准化**: 利用 Box-Cox 转换和标准化优化数据分布和可扩展性
* **平衡目标变量实体**: LLM 推荐的方法如随机过采样、SMOTE 和 ADASYN 帮助平衡数据集，对于无偏见模型训练至关重要
* **数据集划分比例**: LLM 确定数据集的比例（也可以手动调整）
* **模型选择和训练**: LLM 根据数据推荐并使用最适合的模型进行训练
* **群集数量推荐**: 对于聚类任务，使用肘部法则和轮廓系数推荐最佳群集数量（可手动调整）

- 所有处理过的数据和模型都可供下载

### 建模和结果可视化:

![Screenshot 2024-02-12 at 16 10 35](https://github.com/Wilson-ZheLin/Streamline-Analyst/assets/145169519/423da7be-63f1-491d-9ebe-6a788c440c40)

### 自动化工作流界面:

![Screenshot 2024-02-12 at 16 20 19](https://github.com/Wilson-ZheLin/Streamline-Analyst/assets/145169519/9d04d5f2-4f2a-44eb-ab8b-c07c8c0c5a53)

### 支持的建模任务：

| **分类模型**                      | **聚类模型**                   | **回归模型**                         |
|----------------------------------|-------------------------------|-------------------------------------|
| 逻辑回归                          | K-均值聚类                    | 线性回归                             |
| 随机森林                          | DBSCAN                        | 岭回归                               |
| 支持向量机                        | 高斯混合模型                  | Lasso回归                            |
| 梯度提升机                        | 层次聚类                      | 弹性网回归                           |
| 高斯朴素贝叶斯                    | 谱聚类                        | 随机森林回归                         |
| AdaBoost                          | 其他                          | 梯度提升回归                         |
| XGBoost                           |                               | 其他                                 |

### 实时计算模型指标与结果可视化：

| **分类指标 & 图表**                | **聚类指标 & 图表**            | **回归指标 & 图表**                   |
|------------------------------------|--------------------------------|---------------------------------------|
| 模型分数                            | 轮廓分数                        | R平方分数                             |
| 混淆矩阵                            | Calinski-Harabasz 分数         | 均方误差 (MSE)                        |
| AUC                                 | Davies-Bouldin 分数            | 均方根误差 (RMSE)                     |
| F1 分数                             | 聚类散点图                      | 绝对误差 (MAE)                        |
| ROC 曲线                            | 其他                           | 残差图                                |
| 其他                                |                                | 预测值 vs 实际值图                    |
|                                    |                                | 分位数-分位数图                       |


### 可视化分析工具包:

Streamline Analyst 🪄 提供了一系列直观的可视化工具，这部分的使用**无需 API Key**：

* **单属性可视化**: 深入个别数据方面的洞察视图
* **单属性可视化**: 变量间关系的全面分析
* **三维绘图**: 复杂数据关系的3D可视化
* **Word Clouds**: 通过词频突出关键主题和概念
* **世界热力图**: 使地理趋势和分布可视化
* 更多图表正在开发中...


本地运行安装
----------

### 环境&前置准备

运行 `app.py`, 首先需要:
* [Python 3.11.5](https://www.python.org/downloads/)
* [OpenAI API Key](https://openai.com/blog/openai-api)
    * OpenAI: 注意免费API的额度可能不支持GPT-4模型
    
### 安装和运行
1. 安装所需的依赖包

```
pip install -r requirements.txt
```

2. 在本地运行 `app.py`

```
streamlit run app.py
```
