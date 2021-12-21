import streamlit as st
import pandas as pd
import numpy as np
import cv2 as cv
from yolo_image import detect

st.title('Object Detector with YOLOv4')

st.sidebar.title('Object Detector Sidebar')
st.sidebar.subheader('Pages')

app_mode = st.sidebar.selectbox('Select one', ['Home', 'Try it out'])

if app_mode == 'Home':
    st.markdown('This is an object detector that uses the YOLOv4 algorithm')
    st.video('https://www.youtube.com/watch?v=ag3DLKsl2vk&t=114s')

elif app_mode == 'Try it out':
    view_mode = st.radio('Select an Option:', ['Image', 'Webcam'])

    if view_mode == 'Image':
        imageFile = st.sidebar.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
        if imageFile is not None:
            file_bytes = np.asarray(bytearray(imageFile.read()), dtype=np.uint8)
            image = cv.imdecode(file_bytes, 1)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            st.sidebar.text('Original Image')
            st.sidebar.image(image)

            # Detection
            detected_image = detect(image)

            st.subheader('Output Image')
            st.image(detected_image, use_column_width=True)
        
    elif view_mode == 'Webcam':
        run = st.checkbox('Run')
        FRAME_WINDOW = st.image([])
        video = cv.VideoCapture(0)

        while run:
            ret, frame = video.read()
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            detected_frame = detect(frame)
            FRAME_WINDOW.image(detected_frame)
        else:
            pass
    
    else:
        pass

else:
    pass