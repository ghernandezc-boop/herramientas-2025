import streamlit as st
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ------------------ MODELO ------------------
class Classifier(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)

# Cargar modelo
MODEL_PATH = "bmw_classifier.pth"
model = Classifier(input_dim=9)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# ------------------ CODIFICADORES ------------------
# Ajusta estos valores a los que usaste en el entrenamiento
modelos = ['3 Series', '5 Series', '7 Series', 'M3', 'M5', 'X1', 'X3', 'X5', 'X6', 'i3', 'i8']
regiones = ['Africa', 'Asia', 'Europe', 'Middle East', 'North America', 'South America']
colores = ['Black', 'Blue', 'Grey', 'Red', 'Silver', 'White']
combustibles = ['Diesel', 'Electric', 'Hybrid', 'Petrol']
transmisiones = ['Automatic', 'Manual']

# ------------------ APP ------------------
st.title("📊 BMW Sales Classifier")
st.write("Ingresa los datos del auto y obtén la predicción **High / Low** ventas.")

with st.form("auto_form"):
    model_sel = st.selectbox("Modelo", modelos)
    year = st.number_input("Año", min_value=2010, max_value=2025, step=1)
    region_sel = st.selectbox("Región", regiones)
    color_sel = st.selectbox("Color", colores)
    fuel_sel = st.selectbox("Tipo de combustible", combustibles)
    trans_sel = st.selectbox("Transmisión", transmisiones)
    engine = st.number_input("Tamaño del motor (L)", min_value=0.5, max_value=6.0, step=0.1)
    mileage = st.number_input("Kilometraje", min_value=0, max_value=300000, step=1000)
    sales_vol = st.number_input("Volumen de ventas", min_value=100, max_value=10000, step=100)

    submitted = st.form_submit_button("Predecir")

if submitted:
    # Codificación manual (igual que en entrenamiento)
    row = [
        modelos.index(model_sel),
        year - 2010,
        regiones.index(region_sel),
        colores.index(color_sel),
        combustibles.index(fuel_sel),
        transmisiones.index(trans_sel),
        engine,
        mileage,
        sales_vol
    ]
    input_tensor = torch.tensor([row], dtype=torch.float32)
    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.softmax(logits, dim=1)[0]
        pred_class = logits.argmax(dim=1).item()
        label = "High" if pred_class == 1 else "Low"
        confidence = prob[pred_class].item()

    st.success(f"**Predicción:** {label}")  
    st.info(f"**Confianza:** {confidence:.1%}")