import tensorflow as tf
import streamlit as st
import numpy as np
from keras.models import load_model
import cv2
model = load_model("plant_disease_prediction.hdf5")
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
st.title("Plant leaf disease detection")
st.markdown("Upload an image of the plant leaf")

plant_image = st.file_uploader("Choose an image" , type = "jpeg")
submit = st.button("predict Disesase")

if submit:
    if plant_image is not None:
        file_bytes = np.asarray(bytearray(plant_image.read()),dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes,1)
        st.image(opencv_image , channels="RGB")
        st.write(opencv_image.shape)
        opencv_image=cv2.resize(opencv_image,(256,256))
        opencv_image.shape=(1,256,256,3)
        pred = model.predict(opencv_image)
        result = class_names[np.argmax(pred)]
        st.title(result)
    
