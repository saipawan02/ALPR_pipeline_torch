import time
from sklearn.cluster import DBSCAN
import torch
import sys
import os
from paddleocr import PaddleOCR, draw_ocr
import cv2
import json
import datetime
import numpy as np
import re

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

from yolov5 import train, val, detect, export, YOLOv5


def ocrFunc(image_path, ocr):
    output = []
    st = ""
    avg = 0
    dic = {'O': 0, 'o': 0, 'A': 4, 'J': 3, 'l': 1, 'L': 1, 'i': 1, 'I': 1, 'S': 5, 'G': 6, 'H': 8, 'D': 0, 'B': 8,
           'q': 9, 'e': 8, 'b': 6}
    result = ocr.ocr(image_path, cls=True)
    for line in result:
        output.append(line[-1])
        st += line[-1][0]
        avg += line[-1][1]
    if len(result) != 0:
        avg = avg / len(result)

    st = re.sub('[\W_]+', '', st)
    if len(st) < 9 or len(st) > 13:
        st = ""
    return st, avg


def licenseDetectionFunc(license_model, maskedFrame):
    license_results = license_model.predict(maskedFrame)

    # parse results
    predictions = license_results.pred[0]
    boxes = predictions[:, :4].cpu().numpy()  # x1, x2, y1, y2
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

try:
    for f in os.listdir('../numberplates'):
        os.remove(os.path.join('../numberplates', f))
    os.remove("../report/report.json")
except FileNotFoundError:
    print("No files to delete")

model_path = "../yolo/torch/best.pt"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

license_model = YOLOv5(model_path, device)

cap = cv2.VideoCapture("../data/video 3.mp4")
frameRate = cap.get(5)  # frame rate
frameCount = 1

object_detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=100)

ocr = PaddleOCR(use_angle_cls=True, lang='en')

report = {
    'result': []
}

while cap.isOpened():
    frameId = cap.get(1)  # current frame number
    ret, frame = cap.read()

    if not ret:
        break
    height, width, _ = frame.shape

    # frame = frame[height // 2:, ::]

    if frameId % 3 == 0:

        start = time.time()
        mask = object_detector.apply(frame)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        rects = []
        rectsUsed = []
        rect_center = []

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

        license_dic = licenseDetectionFunc(license_model, maskedFrame)

        recognizedTextList = []

        ll = 1
        for box in license_dic['boxes']:
            (x1, y1, x2, y2) = (box[0], box[1], box[2], box[3])
            if (x2 - x1) * (y2 - y1) > 50000:
                continue

            detectedLicenseFileName = '../numberplates/' + str(int(frameCount)) + str(ll) + ".png"

            cv2.imwrite(detectedLicenseFileName, maskedFrame[y1:y2, x1:x2, :])
            text_disp, conf = ocrFunc(detectedLicenseFileName, ocr)

            if len(text_disp) != 0:
                print(text_disp, detectedLicenseFileName)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, text_disp + " " + str(round(conf, 3)), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2,
                            (0, 0, 255), 3)

                report['result'].append({
                    'frame_number': frameCount,
                    'time_stamp':str(datetime.datetime.now()),
                    'license_plate': detectedLicenseFileName,
                    'text': text_disp,
                    'confidence': str(round(conf, 3))
                })

            ll += 1

        elapsed_time = time.time() - start
        fps = 1 / elapsed_time

        # cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 3)
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.imshow("Frame", frame)

        # cv2.imwrite('../frames/currentFrame.jpg', frame)

        key = cv2.waitKey(30)
        if key == 27:
            break
        frameCount += 1

cap.release()
json_object = json.dumps(report, indent=4)
with open("../report/report.json", "w") as outfile:
    outfile.write(json_object)
print("Done!")
