import streamlit as st
import cv2
import numpy as np
from model.preprocess import prepare_image
from model.predict import inference


st.set_page_config(page_title="AI Emotion Detector", page_icon="", layout="centered")
st.title(" Real-time Facial Emotion Recognition")
st.write("This app uses a pre-trained CNN model optimized on the FER2013 dataset to predict human emotions.")

st.markdown("---")

img_file_buffer = st.camera_input("Click 'Take Photo' to analyze your emotion live!")

if img_file_buffer is not None:
    
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    
    processed_frame = prepare_image(cv2_img)
    
    
    emotion, confidence = inference(processed_frame)
    
    
    st.markdown("###   Analysis Results")
    
    if emotion == "No Face Detected":
        st.error(" Please look directly at the camera. No face detected!")
    else:
        st.success(f"Predicted Emotion: **{emotion.upper()}**")
        st.write(f"Confidence Level: {confidence:.2f}%")
        
        st.progress(float(confidence) / 100.0)