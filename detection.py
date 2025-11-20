import cv2 as cv
import numpy as np
from data_processing import detect_signs  

def classify_and_draw(img, model):
    rects = detect_signs(img)
    img2 = img.copy()
    for (x,y,w,h) in rects:
        roi = img[y:y+h, x:x+w]
        roi = cv.resize(roi, (32,32))
        roi = cv.cvtColor(roi, cv.COLOR_BGR2RGB)
        roi = roi / 255.0
        roi = np.expand_dims(roi, 0)
        pred = np.argmax(model.predict(roi))
        color = (0,255,0) if pred != 43 else (0,0,255)
        text = f"Class: {pred}"
        cv.rectangle(img2,(x,y),(x+w,y+h),color,2)
        cv.putText(img2,text,(x,y-10),cv.FONT_HERSHEY_SIMPLEX,0.7,color,2)
    return img2