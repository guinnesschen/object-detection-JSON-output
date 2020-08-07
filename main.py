import time
import base64
import json
import os
import cv2
import numpy as np

CONFIDENCE = 0.5
CONFIG = "yolov3.cfg"
WEIGHTS = "yolov3.weights"
LABELS = "coco.names"

with open(LABELS, 'r') as f:
    labels = f.read().strip().split("\n")

net = cv2.dnn.readNetFromDarknet(CONFIG, WEIGHTS)

def main(encodedImage):
    image_64_decode = base64.decodebytes(encodedImage.encode('utf-8'))

    with open('image.png', 'wb') as image_result:
        image_result.write(image_64_decode)

    image = cv2.imread('image.png')
    os.remove('image.png')
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    h, w = image.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    start = time.perf_counter()
    layer_outputs = net.forward(ln)
    time_took = time.perf_counter() - start
    print(f"Time took: {time_took:.2f}s")

    class_ids = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE:
                class_ids.append(class_id)

    result = []
    for class_id in class_ids:
        result.append(labels[class_id])
    return json.dumps(result)


