from data_processing import load_data
from model import train_model
from detection import classify_and_draw
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np   

if __name__ == "__main__":
    dataset_path = "data/TrainIJCNN2013/"
    X, Y = load_data(dataset_path)
    print("Dataset ROIs =", X.shape, "Classes =", len(np.unique(Y)))
    model, history = train_model(X, Y)
    img = cv.imread("data/TrainIJCNN2013/00005.ppm")
    res = classify_and_draw(img, model)
    plt.figure(figsize=(10,10))
    plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))
    plt.show()