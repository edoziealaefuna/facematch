"""
SnapScan - Flask Backend
Using DeepFace for face recognition (no dlib required)
"""

import os
import base64
import io
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import cv2
from deepface import DeepFace

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

def save_temp_image(image_np, filename):
    """Save numpy array as temp image file for DeepFace"""
    temp_path = f"/tmp/{filename}"
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(temp_path, image_bgr)
    return temp_path

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

    # Blur checks
    blur1_score, blur1_bad, blur1_msg = check_blur(img1)
    blur2_score, blur2_bad, blur2_msg = check_blur(img2)

    if blur1_bad:
        return jsonify({"error": f"Reference image: {blur1_msg}"}), 422
    if blur2_bad:
        return jsonify({"error": f"Comparison image: {blur2_msg}"}), 422

    # Save temp files for DeepFace
    path1 = save_temp_image(img1, "ref.jpg")
    path2 = save_temp_image(img2, "comp.jpg")

    try:
        result = DeepFace.verify(
            img1_path    = path1,
            img2_path    = path2,
            model_name   = "Facenet",
            detector_backend = "opencv",
            enforce_detection = True
        )

        distance  = round(float(result["distance"]), 4)
        threshold = float(result["threshold"])
        is_match  = distance < MATCH_THRESHOLD

        # Get face region sizes
        r1 = result.get("facial_areas", {}).get("img1", {})
        r2 = result.get("facial_areas", {}).get("img2", {})
        w1 = r1.get("w", 0)
        h1 = r1.get("h", 0)
        w2 = r2.get("w", 0)
        h2 = r2.get("h", 0)

        if w1 < MIN_FACE_SIZE or h1 < MIN_FACE_SIZE:
            return jsonify({"error": f"Reference face too small ({w1}x{h1}px). Move closer."}), 422
        if w2 < MIN_FACE_SIZE or h2 < MIN_FACE_SIZE:
            return jsonify({"error": f"Comparison face too small ({w2}x{h2}px). Move closer."}), 422

        return jsonify({
            "match":    is_match,
            "distance": distance,
            "message":  "MATCH" if is_match else "NO MATCH",
            "quality": {
                "ref_blur":   round(blur1_score, 2),
                "comp_blur":  round(blur2_score, 2),
                "ref_size":   f"{w1}x{h1}",
                "comp_size":  f"{w2}x{h2}"
            }
        }), 200

    except ValueError as e:
        return jsonify({"error": "No face detected. Look directly at the camera."}), 422
    except Exception as e:
        return jsonify({"error": f"Face analysis failed: {str(e)}"}), 500

if __name__ == "__main__":
    print("=" * 50)
    print("  SnapScan Flask Server")
    print("  Open: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=False, host="0.0.0.0", port=5000)