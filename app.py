import os
import base64
import io
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import cv2
import face_recognition

app = Flask(__name__)

MATCH_THRESHOLD = 0.45
MIN_FACE_SIZE   = 100
BLUR_THRESHOLD  = 80

def decode_base64_image(b64_string):
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    image_bytes = base64.b64decode(b64_string)
    image_pil   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image_pil)

def check_blur(image_np):
    gray  = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if score < BLUR_THRESHOLD:
        return score, True, f"Image too blurry (score: {score:.1f}). Hold still and ensure good lighting."
    return score, False, "Blur OK"

def check_face_size(face_location):
    top, right, bottom, left = face_location
    width  = right - left
    height = bottom - top
    if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
        return width, height, True, f"Face too small ({width}x{height}px). Move closer to the camera."
    return width, height, False, "Size OK"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/verify", methods=["POST"])
def verify():
    data = request.get_json()
    if not data or "image1" not in data or "image2" not in data:
        return jsonify({"error": "Missing image1 or image2 in request body."}), 400
    try:
        img1 = decode_base64_image(data["image1"])
        img2 = decode_base64_image(data["image2"])
    except Exception as e:
        return jsonify({"error": f"Failed to decode images: {str(e)}"}), 400

    blur1_score, blur1_bad, blur1_msg = check_blur(img1)
    blur2_score, blur2_bad, blur2_msg = check_blur(img2)
    if blur1_bad:
        return jsonify({"error": f"Reference image: {blur1_msg}"}), 422
    if blur2_bad:
        return jsonify({"error": f"Comparison image: {blur2_msg}"}), 422

    locations1 = face_recognition.face_locations(img1, model="hog")
    locations2 = face_recognition.face_locations(img2, model="hog")
    if not locations1:
        return jsonify({"error": "No face detected in reference image. Look directly at the camera."}), 422
    if not locations2:
        return jsonify({"error": "No face detected in comparison image. Look directly at the camera."}), 422

    loc1 = locations1[0]
    loc2 = locations2[0]

    w1, h1, small1, size1_msg = check_face_size(loc1)
    w2, h2, small2, size2_msg = check_face_size(loc2)
    if small1:
        return jsonify({"error": f"Reference image: {size1_msg}"}), 422
    if small2:
        return jsonify({"error": f"Comparison image: {size2_msg}"}), 422

    encodings1 = face_recognition.face_encodings(img1, [loc1])
    encodings2 = face_recognition.face_encodings(img2, [loc2])
    if not encodings1 or not encodings2:
        return jsonify({"error": "Could not generate face encoding. Try better lighting."}), 422

    distance = face_recognition.face_distance([encodings1[0]], encodings2[0])[0]
    is_match  = bool(distance < MATCH_THRESHOLD)

    return jsonify({
        "match":    is_match,
        "distance": round(float(distance), 4),
        "message":  "MATCH" if is_match else "NO MATCH",
        "quality": {
            "ref_blur":   round(blur1_score, 2),
            "comp_blur":  round(blur2_score, 2),
            "ref_size":   f"{w1}x{h1}",
            "comp_size":  f"{w2}x{h2}"
        }
    }), 200

if __name__ == "__main__":
    print("=" * 50)
    print("  FaceMatch Flask Server")
    print("  Open: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)