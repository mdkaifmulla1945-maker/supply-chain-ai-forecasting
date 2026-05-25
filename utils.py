import pandas as pd
import re

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [
        re.sub(r"[^\w]", "_", col) for col in df.columns
    ]

    return df