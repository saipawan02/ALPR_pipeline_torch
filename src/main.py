import cv2
import math
import numpy as np
import time
from sklearn.cluster import DBSCAN
import torch
import sys
import os
import yolov5
import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub
# import easyocr
from paddleocr import PaddleOCR,draw_ocr

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

from yolov5 import train, val, detect, export, YOLOv5


# def preprocess_image(image_path):
#     hr_image = tf.image.decode_image(tf.io.read_file(image_path))
#     if hr_image.shape[-1] == 4:
#         hr_image = hr_image[..., :-1]
#     hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
#     hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
#     hr_image = tf.cast(hr_image, tf.float32)
#     return tf.expand_dims(hr_image, 0)


# def save_image(image, filename):
#     if not isinstance(image, Image.Image):
#         image = tf.clip_by_value(image, 0, 255)
#         image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
#     image.save(filename)


# def superResoluteFunc(maskedFileName, superResolution_model):
#     hr_image = preprocess_image(maskedFileName)

#     fake_image = superResolution_model(hr_image)
#     fake_image = tf.squeeze(fake_image)

#     # SRImgfileName = maskedFileName.split('.')[0]  # + "_Super_Resoluted"

#     save_image(tf.squeeze(fake_image), filename=maskedFileName)

#     return maskedFileName


def ocrFunc(image_path, ocr):
    output = []
    st = ""
    avg = 0
    result = ocr.ocr(image_path, cls=True)
    print(result,image_path)
    for line in result:
      output.append(line[-1])
      st += line[-1][0] 
      avg += line[-1][1] 
    #   print(image_path, line[-1])
    if len(result)!=0:
      avg = avg/len(result)

    return (st,avg) 


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


# vech_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cpu')  # load on CPU
# vech_model.classes = [2, 3, 5, 7]  # 'car', 'motorcycle', 'bus', 'truck'

# set model params for License detection
model_path = "../yolo/torch/best.pt"
device = "cpu"  # or "cpu" or "cuda:0"

# init yolov5 model for license detection
license_model = YOLOv5(model_path, device)

# Vehicle labels
labels_dic = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

# start = time.time()
# main code goes here
cap = cv2.VideoCapture("../data/video 3.mp4")
frameRate = cap.get(5)  # frame rate
frameCount = 1

object_detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=100)

# Super Resolution model weights
# SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

# superResolution_model = hub.load(SAVED_MODEL_PATH)

# easyocr weights
# reader = easyocr.Reader(['en'])

ocr = PaddleOCR(use_angle_cls=True, lang='en') 

while cap.isOpened():

    frameId = cap.get(1)  # current frame number
    ret, frame = cap.read()
    height, width, _ = frame.shape

    # frame = frame[height // 2:, ::]

    if not ret:
        break
    if frameId % 3 == 0:

        start = time.time()
        print(frameCount)

        filename = '../frames/' + str(int(frameCount)) + ".png"
        cv2.imwrite(filename, frame)
        maskedFileName = '../maskedFrames/' + str(int(frameCount)) + ".png"

        mask = object_detector.apply(frame)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_OTSU)
        # mask = cv2.GaussianBlur(mask, (11, 11), 4)
        # kernel = np.ones((5, 5), np.uint8)
        # mask = cv2.dilate(mask, kernel, iterations=10)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        rects = []

        # Bool array indicating which initial bounding rect has
        # already been used
        rectsUsed = []
        rect_center = []
        # Just initialize bounding rects and set all bools to false
        for cnt in contours:
            if cv2.contourArea(cnt) > 1500:
                x, y, w, h = cv2.boundingRect(cnt)

                rect_center.append((int(((x + w // 2) * 25) / width), int(((y + h // 2) * 25) / height)))
                rects.append([x, y, w, h])
        if not rect_center:
            continue
        clustering = DBSCAN(eps=3, min_samples=2).fit(np.array(rect_center))

        final_rect = []
        rect_grps = [[] for i in range(max(clustering.labels_) + 1)]
        for ndx, label in enumerate(clustering.labels_):
            if label == -1:
                final_rect.append(rects[ndx])
            else:
                rect_grps[label].append(rects[ndx])

        for grp in rect_grps:
            if not grp:
                break
            x_min, y_min, x_max, y_max = sys.maxsize, sys.maxsize, 0, 0
            for rec in grp:
                if rec[0] < x_min:
                    x_min = rec[0]
                if rec[1] < y_min:
                    y_min = rec[1]
                if rec[0] + rec[2] > x_max:
                    x_max = rec[0] + rec[2]
                if rec[1] + rec[3] > y_max:
                    y_max = rec[1] + rec[3]
            final_rect.append([x_min, y_min, x_max, y_max])

        acceptedRects = final_rect

        maskedFrame = np.zeros(frame.shape, dtype=float)
        for box in acceptedRects:
            (x1, y1, x2, y2) = (round(box[0]),
                                round(box[1]),
                                round(box[2]),
                                round(box[3]))
            maskedFrame[y1:y2, x1:x2, :] = frame[y1:y2, x1:x2, :]

        # cv2.imshow("Mask", maskedFrame)
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

            # detectedLicenseFileName = superResoluteFunc(detectedLicenseFileName, superResolution_model)
            text_disp,conf = ocrFunc(detectedLicenseFileName, ocr)
            # print(text)
            cv2.putText(frame,text_disp + " " + str(round(conf, 3)), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 3)

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
