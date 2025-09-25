from PIL import Image
import cv2
import torch
import math 
import function.utils_rotate as utils_rotate
from IPython.display import display
import os
import time
import argparse
import function.helper as helper

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=False, default='D:\\2024_2025\\TGMT\\Biensoxe\\License-Plate-Recognition-main\\License-Plate-Recognition-main\\test_image\\video3.mp4', help='path to input image or video')
args = ap.parse_args()

yolo_LP_detect = torch.hub.load('ultralytics/yolov5', 'custom', path='model/LP_detector.pt', force_reload=True)
yolo_license_plate = torch.hub.load('ultralytics/yolov5', 'custom', path='model/LP_ocr.pt', force_reload=True)
yolo_license_plate.conf = 0.60

# Kiểm tra xem đầu vào là video hay ảnh
if args.image.endswith(('.mp4', '.avi', '.mov')):  # Nếu là video
    cap = cv2.VideoCapture(args.image)
    if not cap.isOpened():
        print("Không thể mở video. Kiểm tra đường dẫn.")
        exit()

    while True:
        ret, img = cap.read()
        if not ret:
            break

        plates = yolo_LP_detect(img, size=640)
        list_plates = plates.pandas().xyxy[0].values.tolist()
        list_read_plates = set()

        if len(list_plates) == 0:
            lp = helper.read_plate(yolo_license_plate, img)
            if lp != "unknown":
                cv2.putText(img, lp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                list_read_plates.add(lp)
        else:
            for plate in list_plates:
                flag = 0
                x = int(plate[0])  # xmin
                y = int(plate[1])  # ymin
                w = int(plate[2] - plate[0])  # xmax - xmin
                h = int(plate[3] - plate[1])  # ymax - ymin  
                crop_img = img[y:y+h, x:x+w]
                cv2.rectangle(img, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color=(0,0,225), thickness=2)
                cv2.imwrite("crop.jpg", crop_img)
                rc_image = cv2.imread("crop.jpg")
                lp = ""
                for cc in range(0,2):
                    for ct in range(0,2):
                        lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                        if lp != "unknown":
                            list_read_plates.add(lp)
                            cv2.putText(img, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                            flag = 1
                            break
                    if flag == 1:
                        break

        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

else:  # Nếu là ảnh
    img = cv2.imread(args.image)
    if img is None:
        print("Không thể đọc ảnh. Kiểm tra đường dẫn.")
        exit()

    plates = yolo_LP_detect(img, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    list_read_plates = set()

    if len(list_plates) == 0:
        lp = helper.read_plate(yolo_license_plate, img)
        if lp != "unknown":
            cv2.putText(img, lp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            list_read_plates.add(lp)
    else:
        for plate in list_plates:
            flag = 0
            x = int(plate[0])  # xmin
            y = int(plate[1])  # ymin
            w = int(plate[2] - plate[0])  # xmax - xmin
            h = int(plate[3] - plate[1])  # ymax - ymin  
            crop_img = img[y:y+h, x:x+w]
            cv2.rectangle(img, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color=(0,0,225), thickness=2)
            cv2.imwrite("crop.jpg", crop_img)
            rc_image = cv2.imread("crop.jpg")
            lp = ""
            for cc in range(0,2):
                for ct in range(0,2):
                    lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                    if lp != "unknown":
                        list_read_plates.add(lp)
                        cv2.putText(img, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        flag = 1
                        break
                if flag == 1:
                    break

    cv2.imshow('frame', img)
    cv2.waitKey()
    cv2.destroyAllWindows()