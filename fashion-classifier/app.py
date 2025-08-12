import os
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Get the absolute path to the model file
model_path = os.path.join(os.path.dirname(__file__), 'models', 'fashion-mnist-project.h5')

# Verify model exists before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Load model with error handling
try:
    model = load_model(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    raise

# Class names
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

@app.route('/')
def home():
    return render_template('index.html')

def preprocess_image(image_bytes):
    """Enhanced preprocessing for Fashion-MNIST"""
    img = Image.open(io.BytesIO(image_bytes)).convert('L')  # Grayscale
    img = img.resize((28, 28))
    
    # Convert to numpy array and invert (MNIST-style)
    img_array = 255 - np.array(img)
    
    # Normalize to [0,1] range
    img_array = img_array / 255.0
    
    # Add batch dimension
    return img_array.reshape(1, 28, 28, 1)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Preprocess and predict
        img_array = preprocess_image(file.read())
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        
        return jsonify({
            'class': predicted_class,
            'confidence': confidence,
            'all_predictions': {name: float(score) for name, score in zip(class_names, prediction[0])}
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)