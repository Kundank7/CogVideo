from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import torch
from PIL import Image
import imageio
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load dummy CogVideo model (mock for now)
tokenizer = AutoTokenizer.from_pretrained("THUDM/cogvlm-chat-hf")
model = AutoModelForCausalLM.from_pretrained("THUDM/cogvlm-chat-hf")

@app.route("/")
def home():
    return "CogVideo is running with mock AI!"

@app.route("/generate", methods=["POST"])
def generate_video():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    filename = secure_filename(image_file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(image_path)

    # OPEN IMAGE AND MOCK GENERATE VIDEO (10 frames)
    img = Image.open(image_path).resize((320, 240))
    video_path = os.path.join(OUTPUT_FOLDER, filename.split('.')[0] + ".mp4")

    # Generate mock video by repeating image
    frames = [img for _ in range(10)]
    imageio.mimsave(video_path, frames, fps=2)

    return send_file(video_path, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
