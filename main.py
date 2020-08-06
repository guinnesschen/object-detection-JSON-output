import time
import sys

import cv2
import numpy as np

CONFIDENCE = 0.5
CONFIG = "yolov3.cfg"
WEIGHTS = "yolov3.weights"
LABELS = "coco.names"

with open(LABELS, 'r') as f:
    labels = f.read().strip().split("\n")
    
imagefile = sys.argv[1]
net = cv2.dnn.readNetFromDarknet(CONFIG, WEIGHTS)


def main():
    image = cv2.imread(imagefile)
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    h, w = image.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    start = time.perf_counter()
    layer_outputs = net.forward(ln)
    time_took = time.perf_counter() - start
    print(f"Time took: {time_took:.2f}s")

    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, box_width, box_height) = box.astype("int")
                x = int(centerX - (box_width / 2))
                y = int(centerY - (box_height / 2))
                boxes.append([x, y, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    result = []
    for class_id in class_ids:
        result.append(labels[class_id])
    # print(result)
    new_filename = "objects_in_" + imagefile.split('.')[0] + ".txt"
    with open(new_filename, "w") as f:
        f.write(str(result))


if __name__ == "__main__":
    main()
