from io import StringIO
import pandas as pd


def get_info(df: pd.DataFrame) -> pd.DataFrame:
    info_dict = {
        "Column": df.columns,
        "Non-Null Count": df.count(),
        "Dtype": df.dtypes,
    }
    info_df = pd.DataFrame(info_dict)
    return info_df
