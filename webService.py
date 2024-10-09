from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from PIL import Image
import io
from imageCaptioningModel import loadModel

app = Flask(__name__)
CORS(app)

caption_model = None

def generate_description_from_image(image_path):
    captions = caption_model(img_path=image_path)
    caption_set = set(captions.split(',')) 
    return list(caption_set)

@app.before_request
def load_model_once():
    global caption_model
    caption_model = loadModel()

@app.route('/generate', methods=['POST'])
def generate_description():
    data = request.json
    if 'image_data' not in data:
        return jsonify({"error": "No image data provided"}), 400

    image_data = data['image_data']
    # print("Raw image", image_data)
    image_bytes = base64.b64decode(image_data)

    image_path = "tmp.jpg"
    with open(image_path, "wb") as img_file:
        img_file.write(image_bytes)

    description = generate_description_from_image(image_path)

    return jsonify({"description": description}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
