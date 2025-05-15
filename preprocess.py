import cv2
import imutils

def preprocess(image):
    # Resize
    image = imutils.resize(image, width=300)

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Contrast
    contrast = cv2.equalizeHist(gray)

    # Bilateral filter
    blurred = cv2.bilateralFilter(contrast, 11, 15, 15)
    
    # Threshold
    _, thresh = cv2.threshold(blurred, 112, 255, cv2.THRESH_BINARY)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    #clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Invert
    thresh = cv2.bitwise_not(thresh)
    # Canny
    #canny = cv2.Canny(clean, 120, 255, 1)

    return thresh
