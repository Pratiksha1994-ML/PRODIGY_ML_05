import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load Model
model = tf.keras.models.load_model("models/food_model.h5")

# Dataset Path
dataset_path = r"D:\ML\food\Food Images"

# Class Names
class_names = sorted(os.listdir(dataset_path))

# Calories Dictionary
calorie_dict = {
    "Burger": 295,
    "Cake": 350,
    "Cookies": 502,
    "HotDog": 290,
    "IceCream": 207,
    "PanCakes": 227,
    "Pie": 300,
    "Pizza": 285,
    "Sandwich": 250,
    "Sushi": 200
}

# Title
st.title("Food Recognition & Calorie Estimation")

# Upload Image
uploaded_file = st.file_uploader(
    "Upload Food Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    # Open Image
    img = Image.open(uploaded_file)

    # Show Image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Resize Image
    img = img.resize((224, 224))

    # Convert to Array
    img_array = np.array(img)

    # Normalize
    img_array = img_array / 255.0

    # Expand Dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    # Predicted Class
    predicted_class = class_names[np.argmax(prediction)]

    # Calories
    calories = calorie_dict.get(predicted_class, 0)

    # Display Result
    st.success(f"Food Item: {predicted_class}")

    st.info(f"Estimated Calories: {calories} kcal")