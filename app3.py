from flask import Flask, render_template, request, redirect, url_for
from flask import send_from_directory
import numpy as np
import os
from PIL import Image
import tensorflow as tf  # Import TensorFlow for TensorFlow Lite support

app = Flask(__name__)

# Load the TensorFlow Lite model
MODEL_PATH = 'model_quantized.tflite'
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensors details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set the directory for uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Dog breeds mapping (replace with actual breeds from your model training)
dog_breeds = ['Afghan', 'African Wild Dog', 'Airedale', 'American Hairless', 'American Spaniel',
              'Basenji', 'Basset', 'Beagle', 'Bearded Collie', 'Bermaise', 'Bichon Frise', 
              'Blenheim', 'Bloodhound', 'Bluetick', 'Border Collie', 'Borzoi', 'Boston Terrier', 
              'Boxer', 'Bull Mastiff', 'Bull Terrier', 'Bulldog', 'Cairn', 'Chihuahua', 
              'Chinese Crested', 'Chow', 'Clumber', 'Cockapoo', 'Cocker', 'Collie', 'Corgi', 
              'Coyote', 'Dalmation', 'Dhole', 'Dingo', 'Doberman', 'Elk Hound', 'French Bulldog', 
              'German Shepherd', 'Golden Retriever', 'Great Dane', 'Great Perenees', 'Greyhound', 
              'Groenendael', 'Irish Spaniel', 'Irish Wolfhound', 'Japanese Spaniel', 'Komondor', 
              'Labradoodle', 'Labrador', 'Lhasa', 'Malinois', 'Maltese', 'Mex Hairless', 
              'Newfoundland', 'Pekinese', 'Pit Bull', 'Pomeranian', 'Poodle', 'Pug', 'Rhodesian', 
              'Rottweiler', 'Saint Bernard', 'Schnauzer', 'Scotch Terrier', 'Shar-Pei', 
              'Shiba Inu', 'Shih-Tzu', 'Siberian Husky', 'Vizsla', 'Yorkie']  # Add actual dog breeds

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Preprocess the image for model input
        img = Image.open(filepath)
        img = img.resize((299, 299))  # Resize to match InceptionV3 input size
        img = np.array(img).astype(np.float32) / 255.0  # Normalize image
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Set the tensor as input
        interpreter.set_tensor(input_details[0]['index'], img)

        # Run the interpreter (perform inference)
        interpreter.invoke()

        # Get the prediction result
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        # Extract top 3 predictions
        top_3_indices = np.argsort(predictions)[-3:][::-1]  # Sorting and getting indices
        top_3_breeds = [dog_breeds[i] for i in top_3_indices]
        top_3_confidences_raw = [predictions[i] for i in top_3_indices]

        # Normalize top 3 confidences to sum to 100%
        total_confidence = sum(top_3_confidences_raw)
        top_3_confidences = [(conf / total_confidence) * 100 for conf in top_3_confidences_raw]

        # Zip breeds and confidences
        results = zip(top_3_breeds, top_3_confidences)

        # Pass the zipped results and the image URL to the template
        return render_template('result.html', results=results, image_url=filepath)

    return None


if __name__ == '__main__':
    app.run(debug=True)
