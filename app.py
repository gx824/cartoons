# %%
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np
import random
import logging
import base64
from io import BytesIO

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 生产环境建议使用 INFO 级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://guxiang.pythonanywhere.com"],
        "methods": ["POST", "OPTIONS"]
    }
})

# 修改路径配置，区分本地和服务器环境
if os.environ.get('PYTHONANYWHERE_DOMAIN'):
    UPLOAD_FOLDER = '/tmp/uploads'
    CARTOON_DIR = '/tmp/cartoons'  # 后续建议改为从 GitHub 获取
else:
    UPLOAD_FOLDER = os.path.join('static', 'uploads')
    CARTOON_DIR = os.path.join('static', 'cartoons')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 确保目录存在
def ensure_folders():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(CARTOON_DIR, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 定义30种常见犬种
COMMON_DOG_BREEDS = [
    "Labrador Retriever", "Golden Retriever", "German Shepherd", "Rottweiler",
    "Great Dane", "Saint Bernard", "Doberman Pinscher", "Samoyed",
    "Alaskan Malamute", "Akita", "Border Collie", "Australian Shepherd",
    "Bulldog", "Boxer", "Siberian Husky", "Weimaraner", "Beagle", "Collie",
    "Cocker Spaniel", "Shar Pei", "Bichon Frise", "Yorkshire Terrier",
    "Chihuahua", "Pomeranian", "Dachshund", "Welsh Corgi", "Pug",
    "Shih Tzu", "Poodle", "Mongrel"
]

# 创建犬种映射字典 - 将模型预测的犬种映射到我们的30种常见犬种
BREED_MAPPING = {
    # Labrador相关
    "labrador_retriever": "Labrador Retriever",
    "labrador": "Labrador Retriever",
    "yellow_labrador": "Labrador Retriever",
    "black_labrador": "Labrador Retriever",
    "chocolate_labrador": "Labrador Retriever",

    # Golden Retriever相关
    "golden_retriever": "Golden Retriever",
    "retriever": "Golden Retriever",

    "shiba_inu": "Akita",

    # German Shepherd相关
    "german_shepherd": "German Shepherd",
    "german_shepherd_dog": "German Shepherd",
    "alsatian": "German Shepherd",

    # Rottweiler相关
    "rottweiler": "Rottweiler",

    # Great Dane相关
    "great_dane": "Great Dane",
    "dane": "Great Dane",

    # Saint Bernard相关
    "saint_bernard": "Saint Bernard",
    "st_bernard": "Saint Bernard",

    # Doberman相关
    "doberman": "Doberman Pinscher",
    "doberman_pinscher": "Doberman Pinscher",
    "dobermann": "Doberman Pinscher",

    # Samoyed相关
    "samoyed": "Samoyed",
    "samoyede": "Samoyed",

    # Alaskan Malamute相关
    "alaskan_malamute": "Alaskan Malamute",
    "malamute": "Alaskan Malamute",

    # Akita相关
    "akita": "Akita",
    "japanese_akita": "Akita",
    "american_akita": "Akita",

    # Border Collie相关
    "border_collie": "Border Collie",

    # Australian Shepherd相关
    "australian_shepherd": "Australian Shepherd",
    "aussie_shepherd": "Australian Shepherd",
    "aussie": "Australian Shepherd",

    # Bulldog相关
    "bulldog": "Bulldog",
    "english_bulldog": "Bulldog",
    "american_bulldog": "Bulldog",

    # Boxer相关
    "boxer": "Boxer",

    # Siberian Husky相关
    "siberian_husky": "Siberian Husky",
    "husky": "Siberian Husky",

    # Weimaraner相关
    "weimaraner": "Weimaraner",

    # Beagle相关
    "beagle": "Beagle",

    # Collie相关
    "collie": "Collie",
    "rough_collie": "Collie",
    "smooth_collie": "Collie",

    # Cocker Spaniel相关
    "cocker_spaniel": "Cocker Spaniel",
    "english_cocker_spaniel": "Cocker Spaniel",
    "american_cocker_spaniel": "Cocker Spaniel",

    # Shar Pei相关
    "shar_pei": "Shar Pei",
    "chinese_shar_pei": "Shar Pei",

    # Bichon Frise相关
    "bichon_frise": "Bichon Frise",
    "bichon": "Bichon Frise",

    # Yorkshire Terrier相关
    "yorkshire_terrier": "Yorkshire Terrier",
    "yorkie": "Yorkshire Terrier",

    # Chihuahua相关
    "chihuahua": "Chihuahua",

    # Pomeranian相关
    "pomeranian": "Pomeranian",
    "pom": "Pomeranian",

    # French Bulldog相关
    "french_bulldog": "Bulldog",
    "frenchie": "Bulldog",

    # Welsh Corgi相关
    "welsh_corgi": "Welsh Corgi",
    "pembroke_welsh_corgi": "Welsh Corgi",
    "cardigan_welsh_corgi": "Welsh Corgi",
    "corgi": "Welsh Corgi",

    # Pug相关
    "pug": "Pug",

    # Shih Tzu相关
    "shih_tzu": "Shih Tzu",
    "shih-tzu": "Shih Tzu",

    # Poodle相关
    "toy_poodle": "Poodle",
    "miniature_poodle": "Poodle",
    "standard_poodle": "Poodle",
    "poodle": "Poodle"
}

# 加载模型
model = tf.keras.applications.MobileNetV2(weights='imagenet')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_similar_breed(breed_name):
    """根据名称查找最相似的犬种"""
    # 首先检查精确映射
    if breed_name in BREED_MAPPING:
        return BREED_MAPPING[breed_name]
        
    # 检查是否有部分匹配
    for common_breed in COMMON_DOG_BREEDS:
        common_lower = common_breed.lower()
        key_parts = common_lower.split()
        for part in key_parts:
            if part in breed_name:
                return common_breed
    
    return None

def get_cartoon_image(breed_name, cartoon_dir):
    """从 GitHub 获取卡通图片"""
    import requests
    
    base_url = "https://raw.githubusercontent.com/gx824/cartoons/main/cartoons/"
    image_url = f"{base_url}{breed_name}.png"
    
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            logger.warning(f"未找到{breed_name}的卡通图像，使用随机犬种")
            random_breed = random.choice(COMMON_DOG_BREEDS)
            random_url = f"{base_url}{random_breed}.png"
            response = requests.get(random_url)
            return Image.open(BytesIO(response.content))
    except Exception as e:
        logger.error(f"获取卡通图片失败: {str(e)}")
        raise

# 4. 在第一个请求前创建目录
@app.before_first_request
def create_folders():
    ensure_folders()

@app.route('/')
def index():
    return '''
    <html>
        <head>
            <title>狗狗图片卡通化</title>
            <style>
                body { margin: 40px; font-family: Arial; }
                .container { max-width: 800px; margin: 0 auto; }
                .preview { margin-top: 20px; }
                img { max-width: 100%; margin-top: 10px; }
                .result { margin-top: 20px; }
                .loading { display: none; }
                .error { color: red; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>狗狗图片卡通化</h1>
                <form id="uploadForm">
                    <input type="file" name="file" accept="image/*" required>
                    <input type="submit" value="开始处理">
                </form>
                <div class="loading">处理中...</div>
                <div class="result">
                    <div class="error"></div>
                    <div class="breed-info"></div>
                    <img id="resultImage" style="display: none;">
                </div>
            </div>
            
            <script>
                document.getElementById('uploadForm').onsubmit = async function(e) {
                    e.preventDefault();
                    
                    const loading = document.querySelector('.loading');
                    const error = document.querySelector('.error');
                    const breedInfo = document.querySelector('.breed-info');
                    const resultImage = document.getElementById('resultImage');
                    
                    loading.style.display = 'block';
                    error.textContent = '';
                    breedInfo.textContent = '';
                    resultImage.style.display = 'none';
                    
                    const formData = new FormData(this);
                    
                    try {
                        const response = await fetch('/api/process_image', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const data = await response.json();
                        
                        if (data.error) {
                            error.textContent = data.error;
                        } else {
                            breedInfo.textContent = `识别的犬种: ${data.breed} (置信度: ${(data.confidence * 100).toFixed(2)}%)`;
                            resultImage.src = data.cartoonized_image;
                            resultImage.style.display = 'block';
                        }
                    } catch (err) {
                        error.textContent = '处理失败，请重试';
                    } finally {
                        loading.style.display = 'none';
                    }
                };
            </script>
        </body>
    </html>
    '''
model = None

def load_model():
    global model
    if model is None:
        model = tf.keras.applications.MobileNetV2(weights='imagenet')
    return model

@app.route('/api/process_image', methods=['POST'])
def process_image():
    model = load_model()  # 懒加载模型
    logger.info('接收到新的图片处理请求')
    
    try:
        if 'file' not in request.files:
            logger.error('未检测到上传的图片')
            return jsonify({'error': '请上传图片'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error('文件名为空')
            return jsonify({'error': '请选择文件'}), 400
            
        if file and allowed_file(file.filename):
            # 生成临时文件名并保存
            timestamp = int(time.time())
            filename = secure_filename(f"{timestamp}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                # 保存临时文件
                file.save(filepath)
                
                # 识别犬种
                logger.info('开始犬种识别')
                img = Image.open(filepath).convert('RGB')
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

                predictions = model.predict(img_array)
                decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=10)[0]

                # 记录预测结果
                logger.info('模型预测结果:')
                for _, name, score in decoded:
                    logger.info(f'类别: {name}, 置信度: {score}')

                # 寻找最佳匹配
                best_match = None
                best_confidence = 0

                for _, name, score in decoded:
                    similar_breed = find_similar_breed(name.lower())
                    if similar_breed and score > best_confidence:
                        best_match = similar_breed
                        best_confidence = score

                if not best_match:
                    _, top_name, top_score = decoded[0]
                    best_match = find_similar_breed(top_name.lower())
                    if not best_match:
                        best_match = random.choice(COMMON_DOG_BREEDS)
                    best_confidence = top_score

                # 获取卡通图片
                logger.info(f'识别结果: {best_match}, 置信度: {best_confidence}')
                cartoon_image = get_cartoon_image(best_match, CARTOON_DIR)
                
                # 转换为base64
                buffered = BytesIO()
                cartoon_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # 删除临时文件
                os.remove(filepath)
                
                result = {
                    'success': True,
                    'breed': best_match,
                    'confidence': float(best_confidence),
                    'cartoonized_image': f'data:image/png;base64,{img_str}'
                }
                
                logger.info('处理完成')
                return jsonify(result)
                
            except Exception as e:
                # 确保出错时也删除临时文件
                if os.path.exists(filepath):
                    os.remove(filepath)
                raise e
                
        return jsonify({'error': '不支持的文件类型'}), 400
            
    except Exception as e:
        logger.error(f'处理过程中出错: {str(e)}')
        return jsonify({'error': f'处理失败: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)


