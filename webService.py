from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

def generate_description_from_image(image):
    return "This is a generated description for the uploaded image."

@app.route('/generate', methods=['POST'])
def generate_description():
    data = request.json
    if 'image_data' not in data:
        return jsonify({"error": "No image data provided"}), 400

    image_data = data['image_data']
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))

    description = generate_description_from_image(image)

    return jsonify({"description": description}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
