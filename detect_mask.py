from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 3.0, (224, 224), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detection = faceNet.forward()
    print(detection.shape)

    faces = []
    locs = []
    preds = []

    for i in range(0, detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.5:
            box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w-1, endX), min(h-1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

prototxtPath = r"face_detection/deploy.prototxt"
weightsPath = r"face_detection/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = load_model('mask_detector.model')

print("[INFO] starting video stream...")
vs = VideoStream(scr=0).start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=800)

    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    for (box,pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (maskwearedincorrect, mask, withoutmask) = pred

        if mask > 0.95:
            label = "Mask"
        elif withoutmask > 0.95:
            label = "No Mask"
        else:
            label = "Mask Not Worn Properly"

        if label == "Mask":
            color = (0, 255, 0)
        elif label == "No Mask":
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        label = "{}: {:.2f}%".format(label, max(mask, withoutmask, maskwearedincorrect) * 100)
        cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()
