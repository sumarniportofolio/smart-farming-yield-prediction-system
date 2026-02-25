import streamlit as st
import pandas as pd
from src.utils import model,prepare_input

st.set_page_config(page_title="Smart Farming AI",layout="wide")

st.title("Smart Farming Yield Prediction System")

# sidebar input
st.sidebar.header("Farm Parameters")

soil = st.sidebar.slider("Soil Moisture",0.0,100.0,40.0)
temp = st.sidebar.slider("Temperature",0.0,50.0,25.0)
rain = st.sidebar.slider("Rainfall",0.0,500.0,120.0)
hum = st.sidebar.slider("Humidity",0.0,100.0,60.0)
sun = st.sidebar.slider("Sunlight Hours",0.0,15.0,8.0)
ndvi = st.sidebar.slider("NDVI",0.0,1.0,0.5)
days = st.sidebar.slider("Growth Days",30,180,90)

crop = st.sidebar.selectbox("Crop",["Wheat","Rice","Maize","Cotton","Soybean"])
irrig = st.sidebar.selectbox("Irrigation",["Drip","Sprinkler","Manual","None"])
fert = st.sidebar.selectbox("Fertilizer",["Organic","Inorganic","Mixed"])
disease = st.sidebar.selectbox("Disease",["None","Mild","Moderate","Severe"])

data = {
    "soil_moisture_%":soil,
    "temperature_C":temp,
    "rainfall_mm":rain,
    "humidity_%":hum,
    "sunlight_hours":sun,
    "NDVI_index":ndvi,
    "growth_days":days,
    f"crop_type_{crop}":1,
    f"irrigation_type_{irrig}":1,
    f"fertilizer_type_{fert}":1,
    f"crop_disease_status_{disease}":1
}

df = prepare_input(data)
prediction = model.predict(df)[0]

# metrics row
c1,c2,c3 = st.columns(3)

c1.metric("Predicted Yield",f"{prediction:.2f} kg/ha")
c2.metric("NDVI Score",f"{ndvi*100:.1f}")
c3.metric("Growth Duration",f"{days} days")

# chart
st.subheader("Environmental Summary")

chart = pd.DataFrame({
    "Metric":["Soil","Temp","Rain","Humidity","Sun","NDVI"],
    "Value":[soil,temp,rain,hum,sun,ndvi]
})

st.bar_chart(chart.set_index("Metric"))

# recommendation engine
st.subheader("AI Recommendations")

if soil < 30:
    st.warning("Irigasi disarankan")

if ndvi < 0.4:
    st.error("Tanaman berpotensi stres")

if disease!="None":
    st.warning("Periksa penyakit tanaman")

if soil>70 and rain>300:
    st.info("Risiko overwatering")

st.success("Model aktif")