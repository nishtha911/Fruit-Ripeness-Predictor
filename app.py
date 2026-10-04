import os
import tempfile
import streamlit as st
from PIL import Image
from src.predictor import FruitPredictor

st.set_page_config(
    page_title="Fruit Ripeness & Shelf-Life Predictor",
    page_icon="🍏",
    layout="wide"
)

@st.cache_resource
def load_pipeline():
    return FruitPredictor(
        classifier_path="models/tf_ripeness_model.keras",
        regressor_path="models/shelflife_regressor.pkl"
    )

predictor = load_pipeline()

st.title("🍏 Fruit Ripeness & Shelf-Life Predictor")
st.markdown("Upload a fruit photo and specify storage conditions to predict ripeness and remaining edible days.")

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Step 1: Upload Fruit Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("🌡️ Step 2: Storage Conditions")
    temp = st.slider("Storage Temperature (°C)", min_value=0.0, max_value=40.0, value=20.0, step=0.5)
    humidity = st.slider("Relative Humidity (%)", min_value=20.0, max_value=95.0, value=65.0, step=1.0)

    st.divider()
    
    if uploaded_file is not None:
        if st.button("🔍 Predict Ripeness & Shelf-Life", type="primary", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            with st.spinner("Analyzing image and decay kinetics..."):
                result = predictor.predict(tmp_path, storage_temp_c=temp, humidity_pct=humidity)
                os.remove(tmp_path)

            st.subheader("📊 Prediction Results")
            
            analysis = result.get("ripeness_analysis", result.get("ripenss_analysis", {}))
            stage = analysis.get("ripeness_stage", "Unknown")
            conf = analysis.get("confidence", 0.0) * 100
            days = result["preservation_estimate"]["estimated_shelf_life_days"]

            stage_colors = {"Unripe": "🟢", "Ripe": "🟡", "Overipe": "🔴", "Overripe": "🔴"}
            badge = stage_colors.get(stage, "⚪")

            res_col1, res_col2 = st.columns(2)
            res_col1.metric(label="Ripeness Stage", value=f"{badge} {stage}")
            res_col2.metric(label="Confidence", value=f"{conf:.1f}%")

            st.success(f"⏳ **Estimated Remaining Shelf-Life:** `{days} days` at {temp}°C / {humidity}% humidity")
    else:
        st.info("Please upload an image to run the prediction pipeline.")
