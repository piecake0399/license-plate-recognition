import cv2
import numpy as np
from scipy.ndimage import rotate

"""
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
"""
def preprocess_for_ocr(img):
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while preserving edges
    img = cv2.bilateralFilter(img, 9, 75, 75)

    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Resize image (upscale for better OCR readability)
    img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_CUBIC)

    # Apply Otsu's thresholding (better than adaptive for license plates)
    #_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply light morphological opening (reduces small noise without removing thin strokes)
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Sharpen the image to enhance character distinction
    sharpen_kernel = np.array([[-1, -1, -1], 
                               [-1, 9, -1], 
                               [-1, -1, -1]])
    img = cv2.filter2D(img, -1, sharpen_kernel)

    # Invert
    img = 255 - img
    return img
