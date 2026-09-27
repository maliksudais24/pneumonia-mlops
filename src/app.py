"""
app.py — Flask serving app for the pneumonia detection model.

This wraps our trained model in a small web API. Once running, anyone
(or any system) can send an X-ray image to the /predict endpoint and
get back a prediction — this is the piece that later gets containerized
with Docker and deployed to AWS Lambda.
"""

from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model LAZILY (only when first needed, not at import
# time). This matters for two reasons: (1) it lets CI import and test
# preprocess_image() without needing the full trained model file present,
# and (2) it avoids a slow model load if this module is ever imported
# just for its helper functions.
MODEL_PATH = "pneumonia_model.h5"
_model = None


def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


IMG_SIZE = (224, 224)


def preprocess_image(image_bytes):
    """
    Apply the EXACT same preprocessing used during training:
    resize to 224x224, convert to RGB, normalize to [0,1].
    This consistency is critical — this is exactly the "training/serving
    skew" problem we covered in the feature store section. If preprocessing
    here doesn't match training, predictions will be unreliable.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array


@app.route("/", methods=["GET"])
def index():
    """Serves the demo frontend — the page you actually see in a browser."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Simple health check endpoint — used by orchestrators/load balancers
    to confirm the service is alive."""
    return jsonify({"status": "healthy"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Send an image as 'file'."}), 400

    file = request.files["file"]
    image_bytes = file.read()

    try:
        processed = preprocess_image(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Could not process image: {str(e)}"}), 400

    prediction = get_model().predict(processed)[0][0]  # a value between 0 and 1
    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = float(prediction) if label == "PNEUMONIA" else float(1 - prediction)

    # Structured JSON response — same principle as the structured logging
    # we discussed: easy to parse, easy to log, easy to monitor later.
    response = {
        "prediction": label,
        "confidence": round(confidence, 4),
        "raw_score": round(float(prediction), 4),
    }

    # Simple structured log of every prediction — this is the "prediction
    # telemetry" pillar of observability we covered: logging what was
    # asked and what was predicted, which is what makes drift detection
    # possible later.
    print(f"PREDICTION LOG: {response}")

    return jsonify(response), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)