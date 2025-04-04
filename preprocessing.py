import cv2
import numpy as np
from scipy.ndimage import rotate

def preprocess_for_ocr(img):
    # Grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize
    img_gray = cv2.resize(img_gray, (img_gray.shape[1]*2, img_gray.shape[0]*2), interpolation=cv2.INTER_CUBIC)

    # Threshold
    _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert
    img_invert = 255 - img_thresh

    return img_invert


