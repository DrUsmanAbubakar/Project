from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(12)
ALLOWED_EXTENSIONS = {'jpeg', 'jpg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the pretrained model
model = tf.keras.models.load_model("model/brain_tumor_detection_model.h5")  # Replace with your model path
CLASS_NAMES = ['giloma', 'meningioma', 'notumor', 'pituitary']  # Replace with your tumor class names
print(model.input_shape)
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/scan')
def scan():
    return render_template("scan.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Analyze the image
        result = analyze_image(file_path)
        
        # Return the result to the frontend
        return render_template("result.html", result=result)
    
    flash('Invalid file type')
    return redirect(request.url)

    # if 'file' not in request.files:
    #     flash('No file part')
    #     return redirect(request.url)

    # file = request.files['file']
    # if file.filename == '':
    #     flash('No selected file')
    #     return redirect(request.url)

    # if file and allowed_file(file.filename):
    #     filename = secure_filename(file.filename)
    #     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #     file.save(filepath)

    #     # Analyze the image using the model
    #     prediction = analyze_image(filepath)
    #     return render_template("result.html", tumor_type=prediction)

    # return render_template("scan.html")

def analyze_image(image_path):
    try:
        # Load the image
        img = Image.open(image_path).convert("RGB")  # Ensure the image has 3 channels (RGB)
        
        # Resize to 150x150
        img = img.resize((150, 150))
        
        # Convert to NumPy array and normalize pixel values to [0, 1]
        img_array = np.array(img) / 255.0
        
        # Add batch dimension to match the input shape (None, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Perform prediction
        predictions = model.predict(img_array)
        print("Raw predictions", predictions)
        
        # Get the predicted class
        predicted_class = CLASS_NAMES[np.argmax(predictions)]  # Map output to class names
        return predicted_class
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return "Error processing image"
    try:
        # Open and preprocess the image
        img = Image.open(image_path).convert("RGB")  # Convert to RGB if needed
        img = img.resize((104, 104))  # Resize to match the expected input size
        img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict the tumor type
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        return predicted_class
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return "Error processing image"
    # try:
    # #     # Load and preprocess the image
    # #     img = Image.open(image_path).convert("RGB")
    # #     img = img.resize((128, 128))  # Resize to match model input size
    # #     img_array = np.array(img) / 255.0  # Normalize pixel values
    # #     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # #     # Predict the tumor type
    # #     predictions = model.predict(img_array)
    # #     predicted_class = CLASS_NAMES[np.argmax(predictions)]
    # #     return predicted_class
    # # except Exception as e:
    # #     print(f"Error analyzing image: {e}")
    # #     return "Error processing image"

if __name__ == '__main__':
    app.run(debug=True)
