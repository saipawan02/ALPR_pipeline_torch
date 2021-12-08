import cv2
import math
import numpy as np
import time

import torch
import yolov5

from yolov5 import train, val, detect, export, YOLOv5


def vehicleDetectionFunc(vech_model, filename, frame):
    # declaring masked frame
    maskedFrame = np.zeros(frame.shape, dtype=float)

    vech_results = vech_model(filename)

    # parse results
    predictions = vech_results.pred[0]
    boxes = predictions[:, :4].numpy()  # x1, x2, y1, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    detected_names = []

    for label in categories:
        detected_names.append(labels_dic[int(label)])

    # creating a dict for the vechicle results
    vech_dic = {}

    vech_dic['predictions'] = predictions
    vech_dic['boxes'] = boxes
    vech_dic['scores'] = scores
    vech_dic['categories'] = categories
    vech_dic['categories_names'] = detected_names

    # print('vehicle', vech_dic['categories_names'])

    # initializing masked frame
    for box in boxes:
        (x1, y1, x2, y2) = (round(box[0]),
                            round(box[1]),
                            round(box[2]),
                            round(box[3]))
        maskedFrame[y1:y2, x1:x2, :] = frame[y1:y2, x1:x2, :]

    # save results into "vehicleResults/" folder
    # vech_results.save(save_dir='vehicleResults/')

    return (maskedFrame, vech_dic)


def licenseDetectionFunc(license_model, maskedFileName):
    license_results = license_model.predict(maskedFileName)

    # parse results
    predictions = license_results.pred[0]
    boxes = predictions[:, :4].numpy()  # x1, x2, y1, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    actual_boxes = []

    for box in boxes:
        (x1, y1, x2, y2) = (round(box[0]),
                            round(box[1]),
                            round(box[2]),
                            round(box[3]))
        actual_boxes.append([x1, y1, x2, y2])

    # creating a dict for the license results
    license_dic = {}

    license_dic['predictions'] = predictions
    license_dic['boxes'] = actual_boxes
    license_dic['scores'] = scores
    license_dic['categories'] = categories

    # save results into "licenseResults/" folder
    # license_results.save(save_dir='licenseResults/')

    return license_dic


vech_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cpu')  # load on CPU

# set model params for License detection
model_path = "../yolo/torch/best.pt"
device = "cpu"  # or "cpu" or "cuda:0"

# init yolov5 model for license detection
license_model = YOLOv5(model_path, device)

# Vehicle labels
labels_dic = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

# Selecting only Vehicle classes
vech_model.classes = [2, 3, 5, 7]  # 'car', 'motorcycle', 'bus', 'truck'

# start = time.time()
# main code goes here
cap = cv2.VideoCapture("../data/video 15.mp4")
frameRate = cap.get(5)  # frame rate
frameCount = 1

while cap.isOpened():

    frameId = cap.get(1)  # current frame number
    ret, frame = cap.read()
    height, width, _ = frame.shape

    frame = frame[height // 2:, ::]

    if not ret:
        break
    if frameId % 3 == 0:

        start = time.time()
        print(frameCount)

        filename = '../frames/' + str(int(frameCount)) + ".png"
        cv2.imwrite(filename, frame)
        maskedFileName = '../maskedFrames/' + str(int(frameCount)) + ".png"

        (maskedFrame, vech_dic) = vehicleDetectionFunc(vech_model,
                                                       filename,
                                                       frame)

        # Saving masked Frame
        cv2.imwrite(maskedFileName, maskedFrame)

        license_dic = licenseDetectionFunc(license_model, maskedFileName)

        recognizedTextList = []

        ll = 1
        for box in license_dic['boxes']:
            (x1, y1, x2, y2) = (box[0], box[1], box[2], box[3])

            # Image.fromarray(maskedFrame[y1:y2, x1:x2, :])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            detectedLicenseFileName = '../numberplates/' + str(int(frameCount)) + str(ll) + ".png"

            cv2.imwrite(detectedLicenseFileName,
                        maskedFrame[y1:y2, x1:x2, :])

            ll += 1

        # print('license', license_dic['categories'])

        # print(recognizedTextList)
        elapsed_time = time.time() - start
        fps = 1 / elapsed_time

        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 3)
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(30)
        if key == 27:
            break
        frameCount += 1  # Incrementing frameCount value

cap.release()
print("Done!")
