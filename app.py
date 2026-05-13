import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Smart Waste Classification",
    page_icon="♻️",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(
    model_path="waste_model.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==========================================
# CLASS NAMES
# ==========================================
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

IMG_SIZE = 224

# ==========================================
# TRASH CATEGORY MAP
# ==========================================
trash_map = {
    "eggshells": "Organik",
    "food_waste": "Organik",
    "coffee_grounds": "Organik",
    "tea_bags": "Organik",

    "aerosol_cans": "Anorganik / Logam",
    "aluminum_food_cans": "Anorganik / Logam",
    "aluminum_soda_cans": "Anorganik / Logam",
    "steel_food_cans": "Anorganik / Logam",

    "cardboard_boxes": "Kertas / Kardus",
    "cardboard_packaging": "Kertas / Kardus",
    "magazines": "Kertas",
    "newspaper": "Kertas",
    "office_paper": "Kertas",

    "plastic_soda_bottles": "Plastik",
    "plastic_food_containers": "Plastik",
    "plastic_detergent_bottles": "Plastik",
    "plastic_shopping_bags": "Plastik",
    "plastic_straws": "Plastik",
    "plastic_cup_lids": "Plastik",
    "disposable_plastic_cutlery": "Plastik",

    "glass_beverage_bottles": "Kaca",
    "glass_cosmetic_containers": "Kaca",
    "glass_food_jars": "Kaca",

    "clothing": "Tekstil",
    "shoes": "Tekstil / Karet",

    "styrofoam_cups": "Styrofoam",
    "styrofoam_food_containers": "Styrofoam",

    "paper_cups": "Kertas / Campuran"
}

# ==========================================
# PREDICTION FUNCTION
# ==========================================
def predict_image(image):

    image = image.convert("RGB")

    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image)

    # JANGAN dibagi 255 karena training tidak pakai rescale
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = np.max(prediction) * 100

    return predicted_class, confidence

# ==========================================
# TITLE
# ==========================================
st.title("♻️ Smart Waste Classification Dashboard")

st.write(
    "Dashboard klasifikasi sampah menggunakan model EfficientNetB0."
)

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "📊 Insight Model",
    "🖼️ Prediksi Sampah",
    "📝 Kesimpulan Bisnis"
])

# ==========================================
# TAB 1
# ==========================================
with tab1:

    st.subheader("Hasil Evaluasi Model")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Train Accuracy", "97.41%")
    col2.metric("Validation Accuracy", "84.47%")
    col3.metric("Test Accuracy", "84.35%")
    col4.metric("Test MAE", "0.0122")

    metric_df = pd.DataFrame({
        "Dataset": ["Train", "Validation", "Test"],
        "Accuracy": [97.41, 84.47, 84.35],
        "MAE": [0.0042, 0.0124, 0.0122]
    })

    st.dataframe(metric_df, use_container_width=True)

    st.bar_chart(
        metric_df.set_index("Dataset")[["Accuracy"]]
    )

# ==========================================
# TAB 2
# ==========================================
with tab2:

    st.subheader("Prediksi Jenis Sampah")

    uploaded_file = st.file_uploader(
        "Upload gambar sampah",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Gambar yang diupload",
            use_container_width=True
        )

        with st.spinner("Memprediksi gambar..."):

            predicted_class, confidence = predict_image(image)

        trash_type = trash_map.get(
            predicted_class,
            "Kategori belum tersedia"
        )

        st.success(f"Prediksi: {predicted_class}")

        st.info(f"Confidence: {confidence:.2f}%")

        st.warning(
            f"Rekomendasi tong sampah: {trash_type}"
        )

# ==========================================
# TAB 3
# ==========================================
with tab3:

    st.subheader("Jawaban Pertanyaan Bisnis")

    st.markdown("""
### 1. Seberapa baik performa model?

Model EfficientNetB0 memperoleh:

- Train Accuracy: 97.41%
- Validation Accuracy: 84.47%
- Test Accuracy: 84.35%
- Test MAE: 0.0122

Hasil ini menunjukkan bahwa model mampu melakukan klasifikasi jenis sampah dengan cukup baik dan stabil.

---

### 2. Kategori sampah apa yang paling sulit diklasifikasikan?

Kategori seperti:
- aluminum_food_cans
- steel_food_cans

menjadi kategori yang cukup sulit diprediksi karena memiliki bentuk, warna, dan tekstur yang mirip.

---

### 3. Bagaimana model membantu proses pemilahan sampah?

Model dapat:
- mempercepat identifikasi sampah
- mengurangi human error
- meningkatkan konsistensi klasifikasi
- mendukung smart waste management berbasis AI
""")
