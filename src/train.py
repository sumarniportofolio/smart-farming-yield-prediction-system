import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "dataset.csv")

df = pd.read_csv(data_path)

#df = pd.read_csv("data/dataset.csv", sep = ";")
# =====================
# CLEAN COLUMN NAMES
# =======================
df["sowing_date"] = pd.to_datetime(df["sowing_date"], dayfirst=True, errors="coerce")
df["harvest_date"] = pd.to_datetime(df["harvest_date"], dayfirst=True, errors="coerce")

df["growth_days"] = (df["harvest_date"] - df["sowing_date"]).dt.days

df = df.drop(columns=["sowing_date","harvest_date"])

df = df.dropna()

# =======================
# FEATURE ENGINEERING
# =======================
df["temp_x_humidity"] = df["temperature_C"] * df["humidity_%"]
df["rain_per_day"] = df["rainfall_mm"] / (df["growth_days"]+1)
df["sun_per_day"] = df["sunlight_hours"] / (df["growth_days"]+1)
df["moisture_temp"] = df["soil_moisture_%"] * df["temperature_C"]

# =======================
# ENCODING CATEGORICAL
# =======================
df = pd.get_dummies(df, drop_first=True)

# =======================
# FEATURE SELECTION
# =======================
corr = df.corr(numeric_only=True)["yield_kg_per_hectare"].abs()
selected = corr[corr > 0.02].index
df = df[selected]

# =======================
# SPLIT
# =======================
from sklearn.model_selection import train_test_split

X = df.drop("yield_kg_per_hectare", axis=1)
y = df["yield_kg_per_hectare"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =======================
# SCALING
# =======================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =======================
# MODEL
# =======================
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=400,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =======================
# EVALUATION
# =======================
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)



# =====================
# SAVE MODEL
# =====================
joblib.dump(model, "model_1.pkl")
joblib.dump(X.columns, "columns_1.pkl")

print("\nModel saved successfully.")

#print(df.corr(numeric_only=True)["yield_kg_per_hectare"].abs().sort_values(ascending=False))
