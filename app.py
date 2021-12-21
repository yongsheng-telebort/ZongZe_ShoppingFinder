from os import write
from re import search
import streamlit as st
import pandas as pd
import numpy as np
import cv2 as cv
from streamlit.proto.Button_pb2 import Button

# Phase 2: Define Confidence and NMS Threshold
CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4

# Phase 3: Declare a YOLOv4 model and its labels
# Configure the paths to YOLOv4 files
weights = "yolo-tiny/yolov4-tiny.weights"
labels = "yolo-tiny/labels.txt"
cfg = "yolo-tiny/yolov4-tiny.cfg"
print("You are now using {} as weights ,{} as configs and {} as labels.".format(
    weights, cfg, labels))

# Extract labels from "labels.txt" in yolo folder
lbls = []
with open(labels, "r") as f:
    lbls = [c.strip() for c in f.readlines()]

# Randomly pick a set of colours
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(lbls), 3), dtype="uint8")

# Use OpenCV's built in function to load YoloV4 object detection model, note that cfg and weights need to be configured
net = cv.dnn.readNetFromDarknet(cfg, weights)

# If uses Nvidia GPU
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

# If no GPU
# net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV) h
# net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL_FP16)

# get all the layer names
layer = net.getLayerNames()
test = net.getUnconnectedOutLayers()
layer = [layer[i - 1] for i in test]

# Function to perform object detection


def detect(image):
    nn = net

    # Phase 4: Use YOLOv4 to perform detection
    # Normalize, scale and reshape image to be a suitable input to the neural network
    blob = cv.dnn.blobFromImage(
        image, 1/255, (416, 416), swapRB=True, crop=False)
    print("image.shape:", image.shape)
    print("blob.shape:", blob.shape)

    # Set the blob as input of neural network
    nn.setInput(blob)
    # feed forward (inference) and get the network output
    layer_outs = nn.forward(layer)

    # Extract width and height of image
    (H, W) = image.shape[:2]

    # Check detected objects
    print(len(layer_outs))
    print(layer_outs[0].shape)

    # Phase 4: Obtain Information about the detected objects
    # Declare 3 list to store information about each detected item
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outs:
        for detection in output:
            # Declare scores, class_ids and confidence
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Conditional statement
            if confidence > CONFIDENCE_THRESHOLD:
                box = detection[0:4]
                box = box * np.array([W, H, W, H])
                box = box.astype("int")
                center_x, center_y, width, height = box

                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Phase 5: Draw boxes on image
    idxs = cv.dnn.NMSBoxes(
        boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if len(idxs) > 0:
        for i in idxs.flatten():
            x = boxes[i][0]
            y = boxes[i][1]
            w = boxes[i][2]
            h = boxes[i][3]

            # Select a color based on its labels
            color = [int(c) for c in COLORS[class_ids[i]]]

            # Draw rectangle using openCV
            cv.rectangle(image, (x, y), (x+w, y+h), color, 2)

            text = "{}: {:.2f}%".format(
                lbls[class_ids[i]], confidences[i] * 100)
            cv.putText(image, text, (x, y - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image


st.title("Shopping Finder 🛒🧐")

selected_home = st.sidebar.selectbox(
    "Where you want to go", ["Homepage", "Item detector",
                             "Item Searching", "Feedback", "Support Center"]
)
if selected_home == "Homepage":
    st.header("Homepage")
    st.text(
        "Original Songs(maybe not original owner but the video is original music video)")
    with st.expander("Never gonna give you up (～￣▽￣)～"):
        st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", start_time=0)
    with st.expander("Maneul - Gas gas gas ( •̀ ω •́ )✧"):
        st.video("https://www.youtube.com/watch?v=dzod0j4E-rQ", start_time=0)
    with st.expander("Astronomia(Coffin dance) \(￣︶￣*\))"):
        st.video("https://www.youtube.com/watch?v=j9V78UbdzWI", start_time=0)
    with st.expander("Baka mitai OwO"):
        st.video("https://www.youtube.com/watch?v=e6sBECDG0MU&t=71s", start_time=0)

if selected_home == "Item detector":
    selected_page = st.selectbox(
        "Choose the way to scan the item", ["Detect using image", "Detect using webcam"])
    if selected_page == "Detect using image":
        st.header("Detect using image")
        st.text("Please give me the image so I can detect.")
        imageFile = st.file_uploader(
            'Upload an image', type=['jpg', 'jpeg', 'png'])
        if imageFile is not None:
            file_bytes = np.asarray(
                bytearray(imageFile.read()), dtype=np.uint8)
            image = cv.imdecode(file_bytes, 1)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            st.sidebar.text('Original Image')
            st.sidebar.image(image)

            # Detection
            detected_image = detect(image)

            st.subheader('Output Image')
            st.image(detected_image, use_column_width=True)
        else:
            pass
    elif selected_page == "Detect using webcam":
        st.header("Detect using webcam")
        st.warning('You will be opening your camera so please wear something. 📷🤵')
        run = st.button('Scan')
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
elif selected_home == "Item Searching":
    st.header("Item searching")
    search = st.text_area("What you want to search?")
    if st.button("Search") == True:
        import requests
        import json

        with st.expander("Shopee"):
            url = "https://google-search3.p.rapidapi.com/api/v1/images/q={}+site:shopee.com.my".format(
                search)

            headers = {
                'x-rapidapi-host': "google-search3.p.rapidapi.com",
                'x-rapidapi-key': "6ffd2f508amsh3dd805eadd448cep1f1b09jsn31b5885df010"
            }

            response = requests.request("GET", url, headers=headers)
            result = json.loads(response.text)
            result_image = result["image_results"]
            import re
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            with col1:
                result_src = result_image[0]["image"]["src"]
                st.image(result_src)
                result_link = result_image[0]["link"]["href"]
                result_title = re.sub(
                    '[^a-zA-Z0-9 \n\.]', ' ', result_image[0]["link"]["title"])
                st.markdown(f'''
                [{result_title}]({result_link}) 
                ''')

            with col2:
                result_src = result_image[1]["image"]["src"]
                st.image(result_src)
                result_link = result_image[1]["link"]["href"]
                result_title = re.sub(
                    '[^a-zA-Z0-9 \n\.]', ' ', result_image[1]["link"]["title"])
                st.markdown(f'''
                [{result_title}]({result_link}) 
                ''')
            with col3:
                result_src = result_image[2]["image"]["src"]
                st.image(result_src)
                result_link = result_image[2]["link"]["href"]
                result_title = re.sub(
                    '[^a-zA-Z0-9 \n\.]', ' ', result_image[2]["link"]["title"])
                st.markdown(f'''
                [{result_title}]({result_link}) 
                ''')
            with col4:
                result_src = result_image[3]["image"]["src"]
                st.image(result_src)
                result_link = result_image[3]["link"]["href"]
                result_title = re.sub(
                    '[^a-zA-Z0-9 \n\.]', ' ', result_image[3]["link"]["title"])
                st.markdown(f'''
                [{result_title}]({result_link}) 
                ''')
        with st.expander("Lazada"):

            url = "https://google-search3.p.rapidapi.com/api/v1/images/q={}+site:lazada.com.my".format(
                search)

            headers = {
                'x-rapidapi-host': "google-search3.p.rapidapi.com",
                'x-rapidapi-key': "6ffd2f508amsh3dd805eadd448cep1f1b09jsn31b5885df010"
            }

            response = requests.request("GET", url, headers=headers)
            result = json.loads(response.text)
            result_image = result["image_results"]
            import re

            col5, col6 = st.columns(2)
            col7, col8 = st.columns(2)

            with col5:
                result_src = result_image[4]["image"]["src"]
                st.image(result_src)
                result_link = result_image[4]["link"]["href"]
                result_title = re.sub(
                    '[^a-zA-Z0-9 \n\.]', ' ', result_image[4]["link"]["title"])
                st.markdown(f'''
                [{result_title}]({result_link}) 
                    ''')

            with col6:
                result_src = result_image[1]["image"]["src"]
                st.image(result_src)
                result_link = result_image[1]["link"]["href"]
                result_title = re.sub(
                    '[^a-zA-Z0-9 \n\.]', ' ', result_image[1]["link"]["title"])
                st.markdown(f'''
                [{result_title}]({result_link}) 
                ''')
            with col7:
                result_src = result_image[2]["image"]["src"]
                st.image(result_src)
                result_link = result_image[2]["link"]["href"]
                result_title = re.sub(
                    '[^a-zA-Z0-9 \n\.]', ' ', result_image[2]["link"]["title"])
                st.markdown(f'''
                [{result_title}]({result_link}) 
                ''')
            with col8:
                result_src = result_image[3]["image"]["src"]
                st.image(result_src)
                result_link = result_image[3]["link"]["href"]
                result_title = re.sub(
                    '[^a-zA-Z0-9 \n\.]', ' ', result_image[3]["link"]["title"])
                st.markdown(f'''
                [{result_title}]({result_link}) 
                ''')
        with st.expander("Ebay(Counted with USD)"):

            url = "https://google-search3.p.rapidapi.com/api/v1/images/q={}+site:ebay.com.my".format(
                search)

            headers = {
                'x-rapidapi-host': "google-search3.p.rapidapi.com",
                'x-rapidapi-key': "6ffd2f508amsh3dd805eadd448cep1f1b09jsn31b5885df010"
            }

            response = requests.request("GET", url, headers=headers)
            result = json.loads(response.text)
            result_image = result["image_results"]
            import re

            col9, col10 = st.columns(2)
            col11, col12 = st.columns(2)

            with col9:
                result_src = result_image[4]["image"]["src"]
                st.image(result_src)
                result_link = result_image[4]["link"]["href"]
                result_title = re.sub(
                    '[^a-zA-Z0-9 \n\.]', ' ', result_image[4]["link"]["title"])
                st.markdown(f'''
                [{result_title}]({result_link}) 
                ''')
            with col10:
                result_src = result_image[1]["image"]["src"]
                st.image(result_src)
                result_link = result_image[1]["link"]["href"]
                result_title = re.sub(
                    '[^a-zA-Z0-9 \n\.]', ' ', result_image[1]["link"]["title"])
                st.markdown(f'''
                [{result_title}]({result_link}) 
                ''')
            with col11:
                result_src = result_image[2]["image"]["src"]
                st.image(result_src)
                result_link = result_image[2]["link"]["href"]
                result_title = re.sub(
                    '[^a-zA-Z0-9 \n\.]', ' ', result_image[2]["link"]["title"])
                st.markdown(f'''
                [{result_title}]({result_link}) 
                ''')
            with col12:
                result_src = result_image[3]["image"]["src"]
                st.image(result_src)
                result_link = result_image[3]["link"]["href"]
                result_title = re.sub(
                    '[^a-zA-Z0-9 \n\.]', ' ', result_image[3]["link"]["title"])
                st.markdown(f'''
                [{result_title}]({result_link}) 
                ''')
        with st.expander("Aliexpress"):

            url = "https://google-search3.p.rapidapi.com/api/v1/images/q={}+site:aliexpress.com".format(
                search)

            headers = {
                'x-rapidapi-host': "google-search3.p.rapidapi.com",
                'x-rapidapi-key': "6ffd2f508amsh3dd805eadd448cep1f1b09jsn31b5885df010"
            }

            response = requests.request("GET", url, headers=headers)
            result = json.loads(response.text)
            result_image = result["image_results"]
            import re

            col13, col14 = st.columns(2)
            col15, col16 = st.columns(2)

            with col13:
                result_src = result_image[4]["image"]["src"]
                st.image(result_src)
                result_link = result_image[4]["link"]["href"]
                result_title = re.sub(
                    '[^a-zA-Z0-9 \n\.]', ' ', result_image[4]["link"]["title"])
                st.markdown(f'''
                [{result_title}]({result_link}) 
                ''')
            with col14:
                result_src = result_image[1]["image"]["src"]
                st.image(result_src)
                result_link = result_image[1]["link"]["href"]
                result_title = re.sub(
                    '[^a-zA-Z0-9 \n\.]', ' ', result_image[1]["link"]["title"])
                st.markdown(f'''
                [{result_title}]({result_link}) 
                ''')
            with col15:
                result_src = result_image[2]["image"]["src"]
                st.image(result_src)
                result_link = result_image[2]["link"]["href"]
                result_title = re.sub(
                    '[^a-zA-Z0-9 \n\.]', ' ', result_image[2]["link"]["title"])
                st.markdown(f'''
                [{result_title}]({result_link}) 
                ''')
            with col16:
                result_src = result_image[3]["image"]["src"]
                st.image(result_src)
                result_link = result_image[3]["link"]["href"]
                result_title = re.sub(
                    '[^a-zA-Z0-9 \n\.]', ' ', result_image[3]["link"]["title"])
                st.markdown(f'''
                [{result_title}]({result_link}) 
                ''')
if selected_home == "Feedback":
    st.title("Feedback")
    feedback = st.selectbox("Are our service good enough?", [
                            'Please Choose your answer', 'Yes.', 'Yes,but not the best', 'No'])
    if feedback == 'Yes.':
        st.balloons()
        st.subheader('Thank you for the feedback!')
    elif feedback == 'Yes,but not the best':
        st.subheader(
            'Thank you for using our website,we will improve our website in the future.')
    elif feedback == 'No':
        st.error('101 error you are banned from this website. Rick Astley gonna give you up and say good bye. See this video before you leave ༼ つ ◕_◕ ༽つ. ')
        st.video("https://www.youtube.com/watch?v=vli-PebUUjo")
if selected_home == "Support Center":
    st.title("Support Center")
    st.error("We are having error on loading our page now,please download the rules and regulations below.")
    with open("Rules and Regulations.png", "rb") as file:
        st.download_button(label="Download", data=file,
                           file_name="Rules and Regulations.png")
