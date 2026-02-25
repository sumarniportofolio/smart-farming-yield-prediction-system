import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "dataset.csv")

df = pd.read_csv(data_path)
# =====================
# LOAD
# =====================
#df = pd.read_csv("../data/dataset.csv")

# =====================
# DATE PARSE
# =====================
df["sowing_date"] = pd.to_datetime(df["sowing_date"], format="mixed", dayfirst=True)
df["harvest_date"] = pd.to_datetime(df["harvest_date"], format="mixed", dayfirst=True)

df["growth_days"] = (df["harvest_date"] - df["sowing_date"]).dt.days
df["month"] = df["sowing_date"].dt.month

df.drop(["sowing_date","harvest_date","farm_id","sensor_id"], axis=1, inplace=True)

# =====================
# FEATURE ENGINEERING
# =====================
df["water_index"] = df["rainfall_mm"] * df["soil_moisture_%"]
df["climate_index"] = df["temperature_C"] * df["humidity_%"]
df["sun_water_balance"] = df["sunlight_hours"] / (df["rainfall_mm"] + 1)
df["ndvi_growth"] = df["NDVI_index"] * df["growth_days"]

# =====================
# REMOVE OUTLIER TARGET
# =====================
q1 = df["yield_kg_per_hectare"].quantile(0.01)
q3 = df["yield_kg_per_hectare"].quantile(0.99)
df = df[(df["yield_kg_per_hectare"] > q1) & (df["yield_kg_per_hectare"] < q3)]

# =====================
# SPLIT
# =====================
X = df.drop("yield_kg_per_hectare", axis=1)
y = np.log1p(df["yield_kg_per_hectare"])   # log transform penting

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================
# COLUMN TYPE
# =====================
num_cols = X.select_dtypes(include=["int64","float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# =====================
# PREPROCESSOR
# =====================
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# =====================
# MODEL
# =====================
model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline([
    ("prep", preprocess),
    ("model", model)
])

# =====================
# TRAIN
# =====================
pipe.fit(X_train, y_train)

# =====================
# PREDICT
# =====================
pred_log = pipe.predict(X_test)
pred = np.expm1(pred_log)
y_true = np.expm1(y_test)

# =====================
# METRIC
# =====================
mae = mean_absolute_error(y_true, pred)
rmse = np.sqrt(mean_squared_error(y_true, pred))
r2 = r2_score(y_true, pred)

print("\nRESULT")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

# =====================
# SAVE
# =====================
#joblib.dump(pipe, "../model/model.pkl")