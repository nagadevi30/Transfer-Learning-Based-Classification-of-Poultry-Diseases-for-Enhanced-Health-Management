from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(_name_)

# Load model
model = load_model('model/poultry_model.h5')

# Class mapping (must match folder names in alphabetical order unless set manually)
class_labels = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']

UPLOAD_FOLDER = 'static/images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    if img_file.filename == '':
        return render_template('predict.html', result="No image selected.")

    img_path = os.path.join(UPLOAD_FOLDER, img_file.filename)
    img_file.save(img_path)

    # Preprocess
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    result = class_labels[predicted_index]

    return render_template('predict.html', result=result, image_path=img_path)

if _name_ == '_main_':
    app.run(debug=True)