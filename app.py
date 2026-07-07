import base64
from io import BytesIO
import os
import re
from flask import Flask, jsonify, render_template, request
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, shift
import tensorflow as tf

app = Flask(__name__)
MODEL_PATH = "digit_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)
MIN_DIGIT_PIXELS = 25
MIN_CONFIDENCE = 0.35

def _border_brightness(arr):
    h, w = arr.shape
    margin = max(1, min(h, w) // 20)
    border = np.concatenate(
        [
            arr[0:margin, :].ravel(),
            arr[h - margin : h, :].ravel(),
            arr[:, 0:margin].ravel(),
            arr[:, w - margin : w].ravel(),
        ]
    )
    return float(np.median(border))

def _binarize(arr, is_mnist_style):
    border = _border_brightness(arr)
    if is_mnist_style:
        threshold = border + max(15, (255 - border) * 0.15)
        binary = np.where(arr > threshold, 255, 0).astype(np.uint8)
    else:
        threshold = border - max(15, border * 0.15)
        binary = np.where(arr < threshold, 255, 0).astype(np.uint8)
    h, w = binary.shape
    margin = max(1, min(h, w) // 20)
    edge = np.concatenate(
        [
            binary[0:margin, :].ravel(),
            binary[h - margin : h, :].ravel(),
            binary[:, 0:margin].ravel(),
            binary[:, w - margin : w].ravel(),
        ]
    )
    if float(np.median(edge)) > 127:
        binary = 255 - binary
    return binary

def preprocess_for_mnist(raw_img):
    rgba = raw_img.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    img = Image.alpha_composite(background, rgba).convert("L")
    arr = np.array(img)
    binary = _binarize(arr, is_mnist_style=(_border_brightness(arr) < 80))
    rows = np.any(binary == 255, axis=1)
    cols = np.any(binary == 255, axis=0)
    if not np.any(rows) or not np.any(cols):
        return Image.fromarray(np.zeros((28, 28), dtype=np.uint8))
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    cropped = binary[ymin:ymax+1, xmin:xmax+1]
    cropped_img = Image.fromarray(cropped)
    w, h = cropped_img.size
    if w > h:
        new_w = 20
        new_h = int(round(20.0 * h / w))
        new_h = max(1, new_h)
    else:
        new_h = 20
        new_w = int(round(20.0 * w / h))
        new_w = max(1, new_w)  
    cropped_resized = cropped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas_28 = np.zeros((28, 28), dtype=np.uint8)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    canvas_28[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = np.array(cropped_resized)
    cy, cx = center_of_mass(canvas_28)
    if not np.isnan(cy) and not np.isnan(cx):
        shift_x = 14.0 - cx
        shift_y = 14.0 - cy
        canvas_28 = shift(canvas_28, shift=[shift_y, shift_x], cval=0).astype(np.uint8)
        
    return Image.fromarray(canvas_28)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_data = data["image"]
        image_data = re.sub("^data:image/.+;base64,", "", image_data)
        raw_img = Image.open(BytesIO(base64.b64decode(image_data)))
        img_resized = preprocess_for_mnist(raw_img)
        digit_pixels = int(np.sum(np.array(img_resized) > 127))
        if digit_pixels < MIN_DIGIT_PIXELS:
            return jsonify({"error": "No digit detected. Please draw clearly."}), 400
        img_array = np.array(img_resized, dtype=np.float32).reshape(1, 28, 28, 1) / 255.0
        prediction = model(img_array, training=False).numpy()
        predicted_digit = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))
        if confidence < MIN_CONFIDENCE:
            return jsonify({"error": "Low confidence. Please write more clearly."}), 400
        all_probabilities = {
            str(i): round(float(prob) * 100, 2)
            for i, prob in enumerate(prediction[0])
        }
        return jsonify({
            "digit": predicted_digit,
            "confidence": f"{confidence * 100:.2f}%",
            "all_probabilities": all_probabilities,
        })
    except Exception as e:
        print("An error occurred during processing:", str(e))
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=True)