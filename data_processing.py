import cv2 as cv
import numpy as np
import os
from glob import glob

def detect_signs(img): 
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    red1 = cv.inRange(hsv, (0,80,0), (12,255,255))
    red2 = cv.inRange(hsv, (160,80,0), (180,255,255))
    mask = cv.bitwise_or(red1, red2)
    blur = cv.medianBlur(mask, 9)
    dil = cv.dilate(blur, None, iterations=1)
    contours, _ = cv.findContours(dil, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    H, W = img.shape[:2]
    rects = []
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        if h > H/35 and w > W/35 and h < H/2 and w < W/2:
            rects.append((x,y,w,h))
    return rects

def load_data(dataset_path):
    X = []
    Y = []
    image_paths = sorted(glob(os.path.join(dataset_path, "*.ppm")))
    label_file = os.path.join(dataset_path, "gt.txt")
    labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            d = line.split(';')
            name = d[0]
            cls = int(d[5])
            labels[name] = cls
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        img = cv.imread(img_path)
        if img is None: continue
        rects = detect_signs(img)
        sign_class = labels.get(img_name, 43)
        if len(rects) == 0: continue
        for (x,y,w,h) in rects:
            roi = img[y:y+h, x:x+w]
            roi = cv.resize(roi, (32,32))
            roi = cv.cvtColor(roi, cv.COLOR_BGR2RGB)
            X.append(roi)
            Y.append(sign_class)
    X = np.array(X, dtype="float32") / 255.0
    Y = np.array(Y)
    return X, Y