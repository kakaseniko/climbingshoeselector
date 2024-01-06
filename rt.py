import streamlit as st
import cv2
import torch
from transformers import YolosForObjectDetection, YolosImageProcessor
import helpers
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from skimage.io import imread, imsave


def process_and_display_bounding_boxes(frame, image_placeholder):
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
    inputs = image_processor(images=torch.as_tensor(frame), return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([frame.shape[:2]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        #label_str = model.config.id2label[label.item()]
        #confidence_str = f"{round(score.item(), 3)}"
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        #cv2.putText(frame, f"{label_str}: {confidence_str}", (int(box[0]), int(box[1]) - 10),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    image_placeholder.image(frame, channels="BGR", use_column_width=True, output_format="BGR")
    return results


#def main():
st.title("Real-time Object Detection with YOLOS")
cap = cv2.VideoCapture(2)
#if not cap.isOpened():
#    st.error("Error: Could not open camera.")
#else:
closed = False
image_placeholder = st.empty()
if st.button("Capture Image and Stop"):
    ret, frame = cap.read()
    if ret:
        image_filename = "captured_image.jpeg"
        cv2.imwrite(image_filename, frame)
        
        process_and_display_bounding_boxes(frame, image_placeholder)
        
        st.text(f"Image saved as: {image_filename}")
        closed = True

        cap.release()

while not closed:
    ret, frame = cap.read()
    if not ret:
        #st.error("Error: Could not read frame.")
        break 
    process_and_display_bounding_boxes(frame, image_placeholder)


cap.release()
#find bounding box
footWidth = None
with st.spinner('Processing image...'):
    image_path = './captured_image.jpeg'
    results = helpers.get_bounding_boxes(image_path)
    fig, ax = plt.subplots(1)
    image = Image.open(image_path) 
    ax.imshow(image)
    box = None

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
    st.pyplot(fig)

    #get foot width
    if box:    
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

#get foot shape
#model = tf.keras.models.load_model('footShapeANN80.h5')

REPO_ID = "kakaseniko/fsd"
FILENAME = "fsd.h5"

model = tf.keras.models.load_model(hf_hub_download(repo_id=REPO_ID, filename=FILENAME))
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
resized_image = helpers.resize_image('./edged.jpg', (480, 480))
img = (np.expand_dims(resized_image,0))
prediction = probability_model.predict(img)
footshape = helpers.decodePrediction(prediction)

#get shoes
shoesdf = pd.read_csv('./climbingshoesdata.csv')
if footshape and footWidth:
    shoes = shoesdf.query(f'{footshape} == 1 & {footWidth} == 1')
    st.write(footshape, footWidth)
    st.table(shoes)

    helpers.display_results(shoes)

#if __name__ == "__main__":
#    main()
