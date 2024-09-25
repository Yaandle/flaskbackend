from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from flask_talisman import Talisman
from werkzeug.exceptions import BadRequest
from ultralytics import YOLO
import os, logging, base64, cv2, numpy as np, shutil, zipfile, tempfile
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import pyrealsense2 as rs
import jwt
from jwt.exceptions import InvalidTokenError
from functools import wraps
import requests
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

# Security configurations
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
app.config['MAX_FIELDS'] = 100  # Limit number of form fields

CLERK_JWT_VERIFICATION_KEY = os.environ.get('CLERK_JWT_VERIFICATION_KEY')
CLERK_DOMAIN = os.environ.get('CLERK_DOMAIN')
CLERK_JWT_AUDIENCE = os.environ.get('CLERK_JWT_AUDIENCE')

def get_jwt_key(token):
    jwks_url = f"https://{CLERK_DOMAIN}/.well-known/jwks.json"
    jwks = requests.get(jwks_url).json()
    try:
        kid = jwt.get_unverified_header(token)['kid']
        for key in jwks['keys']:
            if key['kid'] == kid:
                return jwt.algorithms.RSAAlgorithm.from_jwk(key)
    except Exception:
        return None
    
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            token = token.split()[1]  
            key = get_jwt_key(token)
            if not key:
                raise InvalidTokenError('Invalid key')
            jwt.decode(
                token,
                key=key,
                algorithms=["RS256"],
                audience=CLERK_JWT_AUDIENCE,
                issuer=f"https://{CLERK_DOMAIN}",
            )
        except InvalidTokenError as e:
            return jsonify({'message': f'Token is invalid: {str(e)}'}), 401
        return f(*args, **kwargs)
    return decorated

Talisman(app, content_security_policy=None)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
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
            logger.info(f"Loaded model: {name}")
        else:
            logger.error(f"Model file not found at path: {path}")
    return loaded_models

models = initialize_models()
executor = ThreadPoolExecutor(max_workers=4)

@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, InvalidTokenError):
        return jsonify({"error": "Invalid authentication token", "details": str(e)}), 401
    logger.exception("Unhandled exception occurred")
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
@token_required
def process_image():
    logger.info("Starting image processing")
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file"}), 400
        model_name = request.form.get('model')
        mode = request.form.get('task')
        if mode not in ['Detection', 'Segmentation']:
            return jsonify({"error": "Invalid mode"}), 400
        
        model = models.get(model_name)
        if not model:
            return jsonify({"error": f"Invalid model: {model_name}"}), 400
        
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        opencv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if opencv_image is None:
            return jsonify({"error": "Failed to decode image"}), 400
        
        results = model(opencv_image, conf=0.4)
        for result in results:
            annotated_image = result.plot(masks=(mode == 'Segmentation'), labels=True, boxes=(mode == 'Detection'))
        
        if annotated_image is None:
            return jsonify({"error": "No result from model"}), 400
        
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', annotated_image_rgb)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        logger.info("Image processing completed successfully")
        return jsonify({"processedImage": f"data:image/jpeg;base64,{base64_image}"})
    except Exception as e:
        logger.error(f"Error in image processing: {str(e)}", exc_info=True)
        return jsonify({"error": "Image processing failed", "details": str(e)}), 500

@app.route('/ODIS', methods=['POST'])
@limiter.limit("5 per minute")
@token_required
def ODIS():
    logger.info("Received ODIS request")
    if 'files' not in request.files:
        return jsonify({"error": "No files in the request"}), 400
    files = request.files.getlist('files')
    model_name = request.form.get('model')
    if not files:
        return jsonify({"error": "No selected files"}), 400
    
    temp_dir = tempfile.mkdtemp()
    try:
        model = models.get(model_name)
        if not model:
            return jsonify({"error": f"Invalid model: {model_name}"}), 400
        
        total_detections = 0
        saved_images = 0
        no_detection_dir = os.path.join(temp_dir, "No_Detections")
        os.makedirs(no_detection_dir, exist_ok=True)

        for file in files:
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            opencv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if opencv_image is None:
                logger.warning(f"Failed to decode image: {file.filename}")
                continue
            
            results = model(opencv_image, conf=0.4)
            classes_detected = set(result.names[int(box.cls)] for result in results for box in result.boxes)
            total_detections += len(classes_detected)
            
            if not classes_detected:
                logger.warning(f"No objects detected in {file.filename}")
                output_path = os.path.join(no_detection_dir, file.filename)
                if cv2.imwrite(output_path, opencv_image):
                    saved_images += 1
                else:
                    logger.error(f"Failed to save image with no detections: {output_path}")
                continue
            
            for result in results:
                annotated_image = result.plot(masks=False, labels=True, boxes=True)
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                
                for class_name in classes_detected:
                    class_output_dir = os.path.join(temp_dir, class_name)
                    os.makedirs(class_output_dir, exist_ok=True)
                    output_filename = f"{os.path.splitext(file.filename)[0]}_{class_name}.jpg"
                    output_path = os.path.join(class_output_dir, output_filename)
                    if cv2.imwrite(output_path, annotated_image_rgb):
                        saved_images += 1
                    else:
                        logger.error(f"Failed to save image: {output_path}")
        
        logger.info(f"Total detections: {total_detections}, Total saved images: {saved_images}")
        
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    archive_path = os.path.relpath(file_path, temp_dir)
                    zip_file.write(file_path, archive_path)
        
        zip_buffer.seek(0)
        if zip_buffer.getbuffer().nbytes > 0:
            return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='filtered_images.zip')
        else:
            return jsonify({"error": "No objects detected in any of the images"}), 400
    except Exception as e:
        logger.error(f"Error in ODIS: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error in ODIS: {str(e)}"}), 500
    finally:
        shutil.rmtree(temp_dir)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path or 'index.html')

@app.after_request
def add_vary_header(response):
    response.headers["Vary"] = "Cookie"
    return response

def make_verified_request(*args, **kwargs):
    kwargs['verify'] = True
    return requests.request(*args, **kwargs)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)