import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model("plant_disease_model_15.h5")

# Class labels
class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

st.title("Plant Disease Detector (15 Classes)")
st.write("Be a doctor for your Pepper, Potato or Tomato plant!")

uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_resized = cv2.resize(img, (128, 128))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    # Prediction
    preds = model.predict(img_input)
    label = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    st.image(img, channels="BGR", caption="Uploaded Image", use_container_width=True)
    st.success(f"Prediction: {label} ({confidence:.2f}%)")
