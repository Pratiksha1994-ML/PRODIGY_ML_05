import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
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

# Test Image Path
img_path = "test.jpg"

# Load Image
img = image.load_img(img_path, target_size=(224, 224))

# Convert Image to Array
img_array = image.img_to_array(img)

# Expand Dimensions
img_array = np.expand_dims(img_array, axis=0)

# Normalize
img_array = img_array / 255.0

# Prediction
prediction = model.predict(img_array)

# Predicted Class
predicted_class = class_names[np.argmax(prediction)]

# Calories
calories = calorie_dict.get(predicted_class, 0)

# Output
print("Food Item:", predicted_class)
print("Estimated Calories:", calories, "kcal")