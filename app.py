from flask import Flask, request, jsonify, send_file, send_from_directory, current_app
from flask_cors import CORS
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge
from ultralytics import YOLO
import os
import logging
import base64
import cv2
import numpy as np
import shutil
import zipfile
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from logging.handlers import RotatingFileHandler, WatchedFileHandler
import time
import threading
from werkzeug.utils import secure_filename
import io
from PIL import Image
import matplotlib.pyplot as plt

app = Flask(__name__)


app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB
CORS(app, resources={r"/*": {"origins": "*"}})

log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'app.log')

if not app.debug:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    console_handler.setLevel(logging.INFO)
    app.logger.addHandler(console_handler)

app.logger.setLevel(logging.INFO)
app.logger.info('Flask backend startup')

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

MODELS_DIR = os.environ.get('MODELS_DIR', 'UltralyticsModels')
model_paths = {
    'Strawberry': f'{MODELS_DIR}/StrawberryV9.pt',
    'Grapes': f'{MODELS_DIR}/GrapesV1.pt',
    'Apple': f'{MODELS_DIR}/Applev5.pt',
    'YOLOV10': f'{MODELS_DIR}/yolov10m.pt',
    'YOLOV8': f'{MODELS_DIR}/yolov8x.pt',
    'YOLOV5': f'{MODELS_DIR}/yolov5mu.pt',
    'RNF7': f'{MODELS_DIR}/RNF7.pt',
}

def initialize_models():
    loaded_models = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            loaded_models[name] = YOLO(path)
            app.logger.info(f"Loaded model: {name}")
        else:
            app.logger.error(f"Model file not found at path: {path}")
    return loaded_models

models = initialize_models()
executor = ThreadPoolExecutor(max_workers=4)

@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, RequestEntityTooLarge):
        app.logger.warning(f"File too large: {str(e)}")
        return jsonify({"error": "File too large", "details": "The uploaded file exceeds the maximum allowed size."}), 413
    app.logger.exception("Unhandled exception occurred")
    return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/models', methods=['GET'])
def get_models():
    return jsonify([{"name": name, "path": path} for name, path in model_paths.items() if os.path.exists(path)])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

@app.route('/process_image', methods=['POST'])
@limiter.limit("10 per minute")
def process_image():
    app.logger.info("Starting image processing")
    try:
        if 'file' not in request.files:
            app.logger.warning("No file part in request")
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            app.logger.warning(f"Invalid file: {file.filename}")
            return jsonify({"error": "Invalid file"}), 400
        model_name = request.form.get('model')
        mode = request.form.get('task')
        if mode not in ['Detection', 'Segmentation']:
            app.logger.warning(f"Invalid mode: {mode}")
            return jsonify({"error": "Invalid mode"}), 400
        
        model = models.get(model_name)
        if not model:
            app.logger.warning(f"Invalid model: {model_name}")
            return jsonify({"error": f"Invalid model: {model_name}"}), 400
        
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        opencv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if opencv_image is None:
            app.logger.warning("Failed to decode image")
            return jsonify({"error": "Failed to decode image"}), 400
        
        results = model(opencv_image, conf=0.4)
        for result in results:
            annotated_image = result.plot(masks=(mode == 'Segmentation'), labels=True, boxes=(mode == 'Detection'))
        
        if annotated_image is None:
            app.logger.warning("No result from model")
            return jsonify({"error": "No result from model"}), 400
        
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', annotated_image_rgb)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        app.logger.info("Image processing completed successfully")
        return jsonify({"processedImage": f"data:image/jpeg;base64,{base64_image}"})
    except Exception as e:
        app.logger.error(f"Error in image processing: {str(e)}", exc_info=True)
        return jsonify({"error": "Image processing failed", "details": str(e)}), 500

def process_image_odis(file, temp_dir, job_id, model_name):
    try:
        safe_filename = secure_filename(file.filename)
        temp_file_path = os.path.join(temp_dir, safe_filename)
        file.save(temp_file_path)
        model = models[model_name]
        results = model(temp_file_path)

        detected_classes = set()
        for r in results:
            for box in r.boxes:
                class_name = r.names[int(box.cls)]
                detected_classes.add(class_name)
                class_dir = os.path.join(temp_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                try:
                    img = r.plot()
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    base_name = os.path.splitext(safe_filename)[0]
                    annotated_filename = f"{base_name}_{class_name}.jpg"
                    pil_img.save(os.path.join(class_dir, annotated_filename))
                except Exception as e:
                    app.logger.error(f"Failed to save annotated image: {str(e)}")
                shutil.copy2(temp_file_path, os.path.join(class_dir, safe_filename))

        if not detected_classes:
            no_detection_dir = os.path.join(temp_dir, "No_Detections")
            os.makedirs(no_detection_dir, exist_ok=True)
            shutil.copy2(temp_file_path, os.path.join(no_detection_dir, safe_filename))
            app.logger.info(f"No objects detected in {safe_filename}")
        os.remove(temp_file_path)

        return list(detected_classes)

    except Exception as e:
        app.logger.error(f"Error processing image {safe_filename}: {str(e)}", exc_info=True)
        return None


@app.route('/ODIS', methods=['POST'])
def process_odis():
    app.logger.info("Received ODIS request")
    if 'files' not in request.files:
        return jsonify({"error": "No files in the request"}), 400
    
    files = request.files.getlist('files')
    model_name = request.form.get('model')
    if not files:
        return jsonify({"error": "No selected files"}), 400
    if not model_name or model_name not in models:
        return jsonify({"error": f"Invalid model: {model_name}"}), 400
    
    temp_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()  # N
    job_id = str(uuid.uuid4())
    successful_classes = set()
    failed_files = []
    
    try:
        futures = {executor.submit(process_image_odis, file, temp_dir, job_id, model_name): file for file in files}
        for future in as_completed(futures):
            file = futures[future]
            try:
                result = future.result()
                if result:
                    successful_classes.update(result)
                else:
                    failed_files.append(file.filename)
            except Exception as e:
                app.logger.error(f"Error processing {file.filename}: {str(e)}", exc_info=True)
                failed_files.append(file.filename)
        
        # Move processed files to the output directory
        for item in os.listdir(temp_dir):
            shutil.move(os.path.join(temp_dir, item), output_dir)
        
        # Create zip file in a separate directory
        zip_dir = tempfile.mkdtemp()
        output_zip = os.path.join(zip_dir, f"ODIS_results_{job_id}.zip")
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), output_dir))
        
        with open(output_zip, 'rb') as f:
            zip_data = f.read()
        
        # Clean up temporary directories
        shutil.rmtree(temp_dir)
        shutil.rmtree(zip_dir)
        
        response = send_file(
            io.BytesIO(zip_data),
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"ODIS_results_{job_id}.zip"
        )
        
        response.headers['X-Successful-Classes'] = ','.join(successful_classes)
        response.headers['X-Failed-Files'] = ','.join(failed_files)
        
        return response
    
    except Exception as e:
        app.logger.error(f"Error processing ODIS batch: {str(e)}", exc_info=True)
        shutil.rmtree(temp_dir)
        if 'output_dir' in locals():
            shutil.rmtree(output_dir)
        if 'zip_dir' in locals():
            shutil.rmtree(zip_dir)
        return jsonify({"error": "ODIS processing failed", "details": str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/check_zip/<filename>', methods=['GET'])
def check_zip(filename):
    try:
        download_dir = os.path.join(current_app.root_path, 'downloads')
        file_path = os.path.join(download_dir, filename)
        
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            return jsonify({
                "exists": True,
                "size": file_size,
                "unit": "bytes"
            })
        else:
            return jsonify({
                "exists": False
            })
    except Exception as e:
        app.logger.error(f"Error checking zip file {filename}: {str(e)}", exc_info=True)
        return jsonify({"error": "File check failed", "details": str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Server is running"}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
else:
    print("Flask app is being initialized (not running directly)")
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    app.logger.info('Flask app initialized with Gunicorn')
