import os
import io
import base64
from flask import Flask, jsonify, request, render_template, send_from_directory
import cv2
import numpy as np
from flask_cors import CORS
from segment_anything import sam_model_registry, SamPredictor

import pinecone
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image

app = Flask(__name__, template_folder='static')
CORS(app,resources={"*": {"origins": "*"}})

model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def get_image_feature_vector(file):
    img = Image.open(file)
    img = img.resize((500, 500))  # 调整图像大小为模型所需尺寸
    img = img.convert('RGB')
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features.tolist()

# def get_image_feature_vector_demo(base64file):
#     image_data = base64.b64decode(base64file.split(',')[1])
#     file = io.BytesIO(image_data)
#     img = Image.open(file)
#     img = img.resize((500, 500))  # 调整图像大小为模型所需尺寸
#     img = img.convert('RGB')
#     img_array = np.array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)
#     features = model.predict(img_array)
#     flattened_features = features.flatten()
#     normalized_features = flattened_features / np.linalg.norm(flattened_features)
#     return normalized_features.tolist()

# 创立Pinecone的链接
pinecone.init(api_key="54ecd108-25af-49ea-a67a-dbbb465cd177",
              environment="us-west4-gcp")
index_name = "kone"
index = pinecone.Index(index_name)

@app.route('/')
def staticIndex():
    return render_template('index.html')

# 图片路由
@app.route('/uploads/<path:filename>')
def serve_image(filename):
    return send_from_directory('uploads', filename)

# npy路由
@app.route('/npy/<path:filename>')
def serve_npy(filename):
    return send_from_directory('npy', filename)

# 特征路由
@app.route('/features/<path:filename>')
def serve_features(filename):
    return send_from_directory('features', filename)

# 接收批量上传的文件并且保存文件到指定目录
@app.route('/api/upload-files', methods=['POST'])
def uploadFiles():
    files = request.files.getlist('files')
    for file in files:
        if file.filename != '':
            file.save(os.path.join("uploads", file.filename)) 
    return 'Files uploaded successfully'

# 获取所有上传的图片名
@app.route('/api/get-all-files', methods=['GET'])
def getFileNameList():
    folder_path = "uploads"
    file_list = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file_name))]
    return {"data":file_list,"success":True}


# 生成文件的npy
@app.route('/api/generate-npy', methods=['POST'])
def generateNpy():
    data = request.get_json()
    resourceName = data.get('fileName')
    fileName = resourceName.split('.')[0]

    if os.path.exists(os.path.join('npy', f"{fileName}.npy")):
        return 'npy is exist'

    checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device='cpu')
    predictor = SamPredictor(sam)

    image = cv2.imread(f"uploads/{resourceName}")
    predictor.set_image(image)
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    np.save(f"npy/{fileName}.npy", image_embedding)

    return 'Generate npy successfully'


# 读取上传的文件并转换为向量上传到Pinecone,如果成功则把特征图片保存到本地
# @app.route('/api/upload-feature', methods=['POST'])
# def uploadFeature():
#     files = request.files.getlist('files')
#     for file in files:
#         # 处理每个文件
#         filename = file.filename
#         img_feature = get_image_feature_vector(file)
#         upsert_response = index.upsert(vectors=[{
#             'id': filename,
#             'values': img_feature
#         }], namespace='kone')
#         if upsert_response.upserted_count == 1:
#             file.save(os.path.join("features", filename)) 

#     return 'Features uploaded successfully'

# 获取所有特征的文件名
@app.route('/api/get-all-features', methods=['GET'])
def getFeatureNameList():
    folder_path = "features"
    file_list = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file_name))]
    return {"data":file_list,"success":True}

# 根据ID删除对应特征
@app.route('/api/delete-features', methods=['DELETE'])
def deleteFeatures():
    data = request.get_json()
    idList = data.get('ids')
    index.delete(ids=idList, namespace=index_name)
    for file_name in idList:
        file_path = os.path.join('features', file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
    return {"data":"","success":True}

# 读取上传的文件并转换为向量上传到Pinecone,如果成功则把特征图片保存到本地
@app.route('/api/upload-feature', methods=['POST'])
def uploadFeatureDemo():
    data = request.get_json()
    pictureList = data.get('payload')
    for item in pictureList:
        image_data = base64.b64decode(item.get('data').split(',')[1])
        file = io.BytesIO(image_data)
        img_feature = get_image_feature_vector(file)
        upsert_response = index.upsert(vectors=[{
            'id': item.get('name'),
            'values': img_feature
        }], namespace='kone')
        print(upsert_response)
        if upsert_response.upserted_count == 1:
            Image.open(file).save(os.path.join("features", item.get('name'))) 

    return 'demo successfully'


# 运行应用程序
if __name__ == '__main__':
    app.run(debug=True)