"""
├── examples
│   ├── outing.jpg
│   ├── people.jpg
│   ├── street.jpg
|   ├── video_out.avi
|   ├── video.mp4
├── yolo
│   ├── labels.txt
│   ├── yolov4.cfg
│   ├── yolov4.weights
├── yolo-tiny
│   ├── labels.txt
│   ├── yolov4-tiny.cfg
│   ├── yolov4-tiny.weights
├── yolo_image.py
└── yolo_video.py
 if program cant find yolo folder in main folder it will crash."""

# Phase 1: Import Required Libraries
import cv2 as cv
import numpy as np

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
test = [net.getUnconnectedOutLayers()]
layer = [layer[i[0] - 1] for i in test]

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
