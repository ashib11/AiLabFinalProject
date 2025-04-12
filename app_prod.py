from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import os


app = Flask(__name__)
app.debug = True

CORS(app)


app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)


    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[-1] == 1:
        img_array = np.concatenate([img_array] * 3, axis=-1)

    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)


try:
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pneumonia_model.h5')
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise


@app.route('/')
def home():
    print("Home route accessed")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("\nReceived prediction request")
    try:
        if 'file' not in request.files:
            print("⚠️ No file in request")
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        print(f"📄 File received: {file.filename}")

        if file.filename == '':
            print("⚠️ Empty filename")
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"File saved to: {filepath}")


            processed_img = preprocess_image(filepath)
            print("🖼️ Image preprocessed")

            prediction = model.predict(processed_img)[0][0]
            print(f"🔮 Raw prediction value: {prediction}")

            result = "Pneumonia" if prediction > 0.5 else "Normal"
            confidence = float(prediction if result == "Pneumonia" else 1 - prediction)
            print(f"🎯 Result: {result} (Confidence: {confidence:.2%})")

            return jsonify({
                'success': True,
                'prediction': result,
                'confidence': f"{confidence:.2%}",
                'probability': float(prediction)
            })

        else:
            print("Invalid file type")
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg'}), 400

    except Exception as e:
        print(f"🔥 Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

    finally:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
            print("🧹 Temporary file cleaned up")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
