import streamlit as st
import cv2
import numpy as np
from transformers import YolosForObjectDetection, YolosImageProcessor
import torch
from PIL import Image
from sklearn.cluster import KMeans
from skimage.io import imread, imsave
import tensorflow as tf
from huggingface_hub import hf_hub_download

@st.cache_resource
def load_yolo():
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
    return model, image_processor

@st.cache_resource
def load_ann():
    REPO_ID = "kakaseniko/fsd"
    FILENAME = "fsd.h5"
    model = tf.keras.models.load_model(hf_hub_download(repo_id=REPO_ID, filename=FILENAME))
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    return probability_model


def calculateWidthCategory(boxh, boxw):
    ratio = boxw / boxh
    category = 'narrow'
    if ratio < 0.43:
        category = 'narrow'
    elif ratio < 0.48:
        category = 'medium'
    elif ratio > 0.48:
        category = 'wide'
    return category

def get_bounding_boxes(image_path):
    image = Image.open(image_path)
    model, image_processor = load_yolo()

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
    return results

def kMeans_cluster(img):
    image_2D = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    kmeans = KMeans(n_clusters=2, random_state=0).fit(image_2D)
    clustOut = kmeans.cluster_centers_[kmeans.labels_]
    clustered_3D = clustOut.reshape(img.shape[0], img.shape[1], img.shape[2])
    clusteredImg = np.uint8(clustered_3D*255)
    return clusteredImg

def edgeDetection(clusteredImage):
  edged1 = cv2.Canny(clusteredImage, 0, 255)
  edged = cv2.dilate(edged1, None, iterations=1)
  edged = cv2.erode(edged, None, iterations=1)
  return edged

def resize_image(image_path, target_size):
    with Image.open(image_path) as img:
        width, height = img.size
        aspect_ratio = width / height
        new_width = int(min(target_size[0], target_size[1] * aspect_ratio))
        new_height = int(min(target_size[1], target_size[0] / aspect_ratio))
        resized_img = img.resize((new_width, new_height))
        result = Image.new(img.mode, target_size)
        result.paste(resized_img, ((target_size[0] - new_width) // 2,
                                   (target_size[1] - new_height) // 2))       
        return result
    
def decodePrediction(prediction):
    if np.argmax(prediction) == 0:
        return 'egyptian'
    elif np.argmax(prediction) == 1:
        return 'greek'
    elif np.argmax(prediction) == 2:
        return 'roman'

def display_results(shoes):
    for index, row in shoes.iterrows():
        col1, col2 = st.columns(2)

        with col1:
            st.header(row['Model'])
            st.image(f"{row['Model']}.png")

        with col2:
            st.subheader('Level')
            level_options = ['beginner', 'intermediate', 'advanced']
            selected_levels = [level for level in level_options if row[level] == 1]

            selected_levels = selected_levels if len(selected_levels) > 1 else selected_levels[0]

            st.select_slider('', options=level_options, value=selected_levels, key=f"{index}_level", disabled=True)

            st.subheader('Style')
            style_options = ['slabs', 'overhang']
            selected_styles = [style for style in style_options if row[style] == 1]
            st.checkbox('slab', key=f"{index}_slab", value=('slabs' in selected_styles), disabled=True)
            st.checkbox('overhang', key=f"{index}_overhang", value=('overhang' in selected_styles), disabled=True)

            st.subheader('Environment')
            env_options = ['indoor', 'outdoor']
            selected_envs = [env for env in env_options if row[env] == 1]
            st.checkbox('indoor', key=f"{index}_indoor", value=('indoor' in selected_envs), disabled=True)
            st.checkbox('outdoor', key=f"{index}_outdoor", value=('outdoor' in selected_envs), disabled=True)

def predict_foot_shape(image_path):
    probability_model = load_ann()
    resized_image = resize_image(image_path, (480, 480))
    img = (np.expand_dims(resized_image,0))
    prediction = probability_model.predict(img)
    return decodePrediction(prediction)