# Author: Adrian Rosebrock
# https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/
# USAGE
# python deep_learning_object_detection.py --video images/curious_cat.mov -p MobileNetSSD_deploy.prototxt.txt -m MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=False, help="video source(file, url, etc")
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#a
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
    print("Showing video from webcam 0\nPress q to quit.")
# senao, abre a referencia ao arquivo/url do video
else:
    camera = cv2.VideoCapture(args["video"])
    print("Showing video from {}\nPress q to quit.".format(args["video"]))

while True:
    # pega o atual frame
    (grabbed, image) = camera.read()

    # se estamos vendo um vídeo e não conseguiu abrir o frame
    # então sai do loop
    if args.get("video") and not grabbed:
        break

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    #print("[INFO] computing object detections in {}:".format(imagePath))
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            #print("[INFO] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            test_label = CLASSES[idx]
            if test_label in ['cat']:
                print('Detected cat!')
        # exibe o quadro
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    # se 'q' foi pressionada, para o loop
    if key == ord("q"):
        break

# libera o uso da camera e fecha a janela aberta
camera.release()
cv2.destroyAllWindows()