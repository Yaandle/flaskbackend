from flask import Flask, request, jsonify, send_file, send_from_directory, session
from flask_cors import CORS
from flask_talisman import Talisman
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge
from ultralytics import YOLO
import os, logging, base64, cv2, numpy as np, shutil, zipfile, tempfile, uuid
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import jwt
from jwt.exceptions import InvalidTokenError
from functools import wraps
import requests
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  
app.config['SECRET_KEY'] = os.environ['SECRET_KEY']
CLERK_JWT_VERIFICATION_KEY = os.environ.get('CLERK_JWT_VERIFICATION_KEY')
CLERK_DOMAIN = os.environ.get('CLERK_DOMAIN')
CLERK_JWT_AUDIENCE = os.environ.get('CLERK_JWT_AUDIENCE')
CORS(app, resources={r"/*": {
    "origins": os.environ.get('ALLOWED_ORIGINS', '').split(','),
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"]
}})
Talisman(app, content_security_policy={
    'default-src': "'self'",
    'img-src': "'self' data:",
    'script-src': "'self' 'unsafe-inline'"
})
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
limiter = Limiter(app, key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])

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
    if isinstance(e, RequestEntityTooLarge):
        return jsonify({"error": "File too large", "details": "The uploaded file exceeds the maximum allowed size."}), 413
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

def process_image_odis(file, model, temp_dir, job_id):
    image_bytes = file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    opencv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if opencv_image is None:
        logger.warning(f"Failed to decode image: {file.filename}")
        return None
    
    results = model(opencv_image, conf=0.4)
    classes_detected = set(result.names[int(box.cls)] for result in results for box in result.boxes)
    
    if not classes_detected:
        logger.warning(f"No objects detected in {file.filename}")
        output_path = os.path.join(temp_dir, "No_Detections", file.filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if cv2.imwrite(output_path, opencv_image):
            return "No_Detections"
        else:
            logger.error(f"Failed to save image with no detections: {output_path}")
            return None
    
    for result in results:
        annotated_image = result.plot(masks=False, labels=True, boxes=True)
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        for class_name in classes_detected:
            class_output_dir = os.path.join(temp_dir, class_name)
            os.makedirs(class_output_dir, exist_ok=True)
            output_filename = f"{os.path.splitext(file.filename)[0]}_{class_name}.jpg"
            output_path = os.path.join(class_output_dir, output_filename)
            if cv2.imwrite(output_path, annotated_image_rgb):
                return class_name
            else:
                logger.error(f"Failed to save image: {output_path}")
                return None
    
    return None

@app.route('/ODIS', methods=['POST'])
@limiter.limit("20 per hour")
@token_required  
def ODIS():
    logger.info("Received ODIS request")
    if 'files' not in request.files:
        return jsonify({"error": "No files in the request"}), 400
    
    files = request.files.getlist('files')
    model_name = request.form.get('model')
    if not files:
        return jsonify({"error": "No selected files"}), 400
    if not model_name or model_name not in models:
        return jsonify({"error": f"Invalid model: {model_name}"}), 400
    
    model = models[model_name]
    temp_dir = tempfile.mkdtemp()
    job_id = str(uuid.uuid4())
    successful_classes = []
    failed_files = []
    try:
        futures = {executor.submit(process_image_odis, file, model, temp_dir, job_id): file for file in files}
        for future in as_completed(futures):
            file = futures[future]
            try:
                result = future.result()
                if result:
                    successful_classes.append(result)
                else:
                    failed_files.append(file.filename)
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}", exc_info=True)
                failed_files.append(file.filename)
        
        output_zip = os.path.join(temp_dir, f"ODIS_results_{job_id}.zip")
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), temp_dir))
        
        response_data = {
            "successful_classes": successful_classes,
            "failed_files": failed_files,
            "download_link": f"/download/{os.path.basename(output_zip)}"
        }
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error processing ODIS batch: {str(e)}", exc_info=True)
        return jsonify({"error": "ODIS processing failed", "details": str(e)}), 500
    finally:
        shutil.rmtree(temp_dir)

@app.route('/download/<filename>', methods=['GET'])
@token_required
def download_file(filename):
    try:
        return send_from_directory(tempfile.gettempdir(), filename, as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading file {filename}: {str(e)}", exc_info=True)
        return jsonify({"error": "File download failed", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
