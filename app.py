from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import logging
import base64
import cv2
import numpy as np
from werkzeug.exceptions import BadRequest
from ultralytics import YOLO
import shutil
import zipfile
from io import BytesIO

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = 'UltralyticsModels'
DATASETS_DIR = 'Datasets'

model_paths = {
    'Strawberry': 'UltralyticsModels/StrawberryV9.pt',
    'Grapes': 'UltralyticsModels/GrapesV1.pt',
    'Apple': 'UltralyticsModels/Applev5.pt',
    'YOLOV10': 'UltralyticsModels/yolov10m.pt',
    'YOLOV8': 'UltralyticsModels/yolov8x.pt',
    'YOLOV5': 'UltralyticsModels/yolov5mu.pt',
    'RNF4': 'UltralyticsModels/RiderNF4.pt',
}
model_name = model_paths
@app.route('/models', methods=['GET'])
def get_models():
    models = []
    for model_name, model_path in model_paths.items():
        if os.path.exists(model_path):
            models.append({
                'name': model_name,
                'path': model_path
            })
    logger.info(f"Models fetched: {models}")
    return jsonify(models)

@app.route('/datasets', methods=['GET'])
def get_datasets():
    datasets = []
    if os.path.exists(DATASETS_DIR):
        for dataset in os.listdir(DATASETS_DIR):
            path = os.path.join(DATASETS_DIR, dataset)
            if os.path.isdir(path):
                datasets.append({
                    'name': dataset,
                    'size': sum(os.path.getsize(os.path.join(dirpath, filename)) 
                                for dirpath, dirnames, filenames in os.walk(path) 
                                for filename in filenames),
                    'updated': os.path.getmtime(path),
                    'path': path
                })
    logger.info(f"Datasets fetched: {datasets}")
    return jsonify(datasets)

@app.route('/download', methods=['POST'])
def download_file():
    data = request.json
    if not data or 'path' not in data:
        return jsonify({'error': 'No path provided'}), 400
    path = data['path']
    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
    try:
        return send_file(path, as_attachment=True)
    except Exception as e:
        logger.error(f"Error sending file: {str(e)}", exc_info=True)
        return jsonify({'error': 'Error sending file'}), 500

models = {}
for name, path in model_paths.items():
    if os.path.exists(path):
        models[name] = YOLO(path)
        logger.info(f"Loaded model: {name}")
    else:
        logger.error(f"Model file not found at path: {path}")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/process_image', methods=['POST'])
def process_image():
    logger.info("Received image processing request")
    if 'file' not in request.files:
        raise BadRequest("No file part in the request")
    file = request.files['file']
    model_name = request.form.get('model')
    mode = request.form.get('task')
    logger.info(f"Processing image with model: {model_name}, mode: {mode}")
    if file.filename == '':
        raise BadRequest("No selected file")
    if mode not in ['Detection', 'Segmentation']:
        raise BadRequest(f"Invalid mode: {mode}")
    try:
        model = models.get(model_name, None)
        if not model:
            raise BadRequest(f"Model not loaded or invalid model name: {model_name}")
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        opencv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if opencv_image is None:
            raise Exception("Failed to decode image")
        logger.info(f"Image shape is: {opencv_image.shape}")
        results = model(opencv_image, conf=0.4)
        annotated_image = None
        for result in results:
            if mode == 'Detection':
                annotated_image = result.plot(masks=False, labels=True, boxes=True)
            elif mode == 'Segmentation':
                annotated_image = result.plot(masks=True, labels=True, boxes=False)
        if annotated_image is None:
            raise Exception("No result from model")
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', annotated_image_rgb)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        logger.info("Image processed successfully")
        return jsonify({"processedImage": f"data:image/jpeg;base64,{base64_image}"})
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path or 'index.html')

@app.route('/ODIS', methods=['POST'])
def ODIS():
    logger.info("Received ODIS request")
    if 'files' not in request.files:
        raise BadRequest("No file part in the request")
    files = request.files.getlist('files')
    model_name = request.form.get('model')
    task = request.form.get('task')
    logger.info(f"Running ODIS with model: {model_name}, task: {task}")
    if not files:
        raise BadRequest("No selected files")
    if task != 'Detection':
        raise BadRequest(f"Invalid task: {task}. Only 'Detection' is supported.")
    temp_dir = 'temp_odis_results'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    try:
        model = models.get(model_name)
        if not model:
            raise BadRequest(f"Model not loaded or invalid model name: {model_name}")
        for file in files:
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            opencv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if opencv_image is None:
                raise Exception(f"Failed to decode image: {file.filename}")
            results = model(opencv_image, conf=0.4)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_name = result.names[int(box.cls)]
                    output_dir = os.path.join(temp_dir, class_name)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    annotated_image = result.plot(masks=False, labels=True, boxes=True)
                    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    output_filename = f"{file.filename.split('.')[0]}_{class_name}.jpg"
                    output_path = os.path.join(output_dir, output_filename)
                    cv2.imwrite(output_path, annotated_image_rgb)
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    zip_file.write(os.path.join(root, file), 
                                   os.path.relpath(os.path.join(root, file), 
                                   temp_dir))
        zip_buffer.seek(0)
        shutil.rmtree(temp_dir)
        return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='filtered_images.zip')
    except Exception as e:
        logger.error(f"Error in ODIS: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error in ODIS: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)