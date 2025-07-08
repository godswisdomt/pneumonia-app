from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = load_model("pneumonia_cnn_model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return "No file uploaded", 400

    try:
        # Read and preprocess the image from memory
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize((150, 150))
        image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

        prediction = model.predict(image_array)
        result = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"

        return render_template('result.html', prediction_text=result)

    except Exception as e:
        return f"❌ Error processing image: {e}"

if __name__ == '__main__':
    app.run(debug=True)
