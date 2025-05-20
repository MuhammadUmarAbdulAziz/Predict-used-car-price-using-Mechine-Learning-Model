from pathlib import Path
import pandas as pd
import streamlit as st
import predict

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 24px;
    border-radius: 8px;
    font-size: 16px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🚗 USED-CAR PRICE PREDICTION</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Estimate your car's price with Machine Learning!</p>", unsafe_allow_html=True)

# --- IMAGE ---
image_path = str(Path(__file__).parents[1] / 'used-car-all-brand-sell--613.jpg')
st.image(image_path, use_column_width=True)

st.markdown("---")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    return pd.read_csv(str(Path(__file__).parents[1] / 'used_car_cleaned.csv'))

used_car = load_data()

# --- USER INPUT FUNCTION ---
def get_user_input(df):
    col1, col2 = st.columns(2)

    with col1:
        car_maker = st.selectbox("Manufacturer:", df['Make'].unique(), key="maker")
        car_type = st.selectbox("Model/Type:", df[df['Make'] == car_maker]['Type'].unique(), key="type")
        car_origin = st.selectbox("Origin/From:", df['Origin'].unique(), key="origin")
        car_region = st.selectbox("Selling/Buying Region:", df['Region'].unique(), key="region")
        car_gear_type = st.selectbox("Gear Type:", df['Gear_Type'].unique(), key="gear")

    with col2:
        car_option = st.selectbox("Car's Option:", df['Options'].unique(), key="option")
        car_year = st.number_input("Year of Production:", value=2010, step=1, key="year")
        car_engine_size = st.number_input("Engine Size (L):", value=1.5, step=0.1, key="engine")
        car_mileage = st.number_input("Mileage (km):", value=0, step=1000, key="mileage")

    user_data = pd.DataFrame({
        'Type': [car_type],
        'Region': [car_region],
        'Make': [car_maker],
        'Gear_Type': [car_gear_type],
        'Origin': [car_origin],
        'Options': [car_option],
        'Year': [car_year],
        'Engine_Size': [car_engine_size],
        'Mileage': [car_mileage]
    })

    return user_data


# --- GET USER INPUT ---
user_data = get_user_input(used_car)

# --- PREDICT BUTTON ---
if st.button("🔮 Predict Price"):
    used_car_price = round(predict.predict(user_data)[0], 2)
    formatted_price = "{:,.2f}".format(used_car_price)

    st.markdown(f"""
    <div style='padding:20px; background-color:#d4edda; border:1px solid #c3e6cb; border-radius:10px; text-align:center;'>
        <h2>Estimated Car Price:</h2>
        <p style='font-size:24px; font-weight:bold; color:#155724;'>R{formatted_price} RSA</p>
        <p>This is an estimate using a Machine Learning model.</p>
    </div>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Created by: Risdan Kristori</p>", unsafe_allow_html=True)