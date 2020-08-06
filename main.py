import cv2, numpy as np, time, sys

config = "yolov3.cfg"
weights = "yolov3.weights"

with open("coco.names") as f:
    labels = f.read().strip().split("\n")
    
imagefile = sys.argv[1]

colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

image = cv2.imread(imagefile)
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
h, w = image.shape[:2]

CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

net = cv2.dnn.readNetFromDarknet(config, weights)

net.setInput(blob)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
start = time.perf_counter()
layer_outputs = net.forward(ln)
time_took = time.perf_counter() - start
print(f"Time took: {time_took:.2f}s")

font_scale = 1
thickness = 1
boxes, confidences, class_ids = [], [], []
for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > CONFIDENCE:
            box = detection[:4] * np.array([w, h, w, h])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)
result = []
for id in class_ids:
    result.append(labels[id])
print(result)
newFileName = "objects_in_" + imagefile.split('.')[0] + ".txt"
f = open(newFileName, "w")
f.write(str(result))
