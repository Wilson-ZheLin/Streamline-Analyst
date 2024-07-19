import os
import pandas as pd
from src.llm_service import select_pipeline
from src.utils.dataframe_util import get_info

PIPELINE_MAP = {
    1: "Data Visualization",
    2: "Predictive Classification",
    3: "Regression Model",
    4: "Data Visualization",
}

DEFAULT_PIPELINE = "Data Visualization"


def get_selected_pipeline(data: pd.DataFrame, question: str, model_type=4, api_key="") -> str:
    return PIPELINE_MAP.get(
        select_pipeline(
            shape_info=data.shape,
            description_info=get_info(data),
            head_info=data.head(10),
            question=question,
            model_type=model_type,
            user_api_key=api_key,
        ),
        DEFAULT_PIPELINE,
    )
