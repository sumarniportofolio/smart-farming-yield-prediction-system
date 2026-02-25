import joblib
import pandas as pd

model = joblib.load("model/model_1.pkl")
cols = joblib.load("model/columns_1.pkl")

def prepare_input(data):
    df = pd.DataFrame([data])

    for c in cols:
        if c not in df.columns:
            df[c]=0

    return df[cols]