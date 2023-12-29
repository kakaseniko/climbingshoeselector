import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.io import imread, imsave
import tensorflow as tf
import pandas as pd
import helpers

@st.cache_data
def load_shoes():
    shoesdf = pd.read_csv('./climbingshoesdata.csv')
    return shoesdf

#APP
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    #take image
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite('captured.jpeg', cv2_img)

    #find bounding box
    image_path = './captured.jpeg'
    results = helpers.get_bounding_boxes(image_path)

    if len(results["boxes"]) == 0:
        st.error("No foot detected")
        st.stop()

    fig, ax = plt.subplots(1)
    image = Image.open(image_path) 
    ax.imshow(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
    st.pyplot(fig)

    #get foot width
    boxw = box[2] - box[0]
    boxh = box[3] - box[1]
    if boxw > boxh:
        boxw = box[3] - box[1]
        boxh = box[2] - box[0] 
    footWidth = helpers.calculateWidthCategory(boxh, boxw)

    #process image
    img = imread(image_path)
    clusteredImage = helpers.kMeans_cluster(img)
    st.image(clusteredImage, caption='Clustered Image', use_column_width=True)

    edgedImg = helpers.edgeDetection(clusteredImage)
    st.image(edgedImg, caption='Edged Image', use_column_width=True)
    imsave('edged.jpg', edgedImg)

    footshape = helpers.predict_foot_shape('./edged.jpg')

    #get shoes
    shoesdf = load_shoes()
    shoes = shoesdf.query(f'{footshape} == 1 & {footWidth} == 1')
    st.write(footshape, footWidth)
    st.table(shoes)

    helpers.display_results(shoes)





