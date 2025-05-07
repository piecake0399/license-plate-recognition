import cv2
import imutils
import numpy as np

def preprocess(image):
    # Resize
    image = imutils.resize(image, width=300)

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Contrast
    contrast = cv2.equalizeHist(gray)

    # Bilateral filter
    blurred = cv2.bilateralFilter(contrast, 11, 17, 17)

    # Thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return blurred
