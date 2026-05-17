import streamlit as st
import numpy as np
import joblib
import os

# Page config
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(BASE_DIR, 'src', 'house_model.pkl'))
    return model

model = load_model()

# Header
st.title("🏠 House Price Predictor")
st.markdown("**Built by Sujitha | AI/ML Portfolio Project**")
st.markdown("---")

# Input fields
st.subheader("🔢 Enter House Details")

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    gr_liv_area = st.number_input("Living Area (sq ft)", 500, 5000, 1500)
    garage_cars = st.selectbox("Garage Cars", [0, 1, 2, 3, 4])
    total_bsmt = st.number_input("Basement Area (sq ft)", 0, 3000, 800)
    full_bath = st.selectbox("Full Bathrooms", [1, 2, 3, 4])

with col2:
    year_built = st.slider("Year Built", 1900, 2024, 2000)
    year_remod = st.slider("Year Remodeled", 1900, 2024, 2005)
    lot_area = st.number_input("Lot Area (sq ft)", 1000, 50000, 8000)
    bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6])
    neighborhood = st.selectbox("Neighborhood Code", list(range(0, 25)))

st.markdown("---")

# Predict button
if st.button("🔮 Predict Price", use_container_width=True):
    features = np.array([[
        overall_qual, gr_liv_area, garage_cars,
        total_bsmt, full_bath, year_built,
        year_remod, lot_area, bedrooms, neighborhood
    ]])

    log_price = model.predict(features)[0]
    price = np.expm1(log_price)

    st.success(f"### 💰 Estimated House Price: ${price:,.0f}")

    low = price * 0.90
    high = price * 1.10
    st.info(f"📊 Likely Range: ${low:,.0f} — ${high:,.0f}")

st.markdown("---")
st.caption("🤖 Powered by Random Forest | Dataset: Kaggle House Prices")