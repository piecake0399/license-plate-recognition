import cv2
import imutils

def preprocess(image):
    # Resize
    image = imutils.resize(image, width=300)
    image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Contrast
    #contrast = cv2.equalizeHist(gray)

    # Bilateral filter
    #blurred = cv2.bilateralFilter(contrast, 11, 15, 15)
    
    # Threshold
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    dilated = cv2.dilate(thresh,kernel,iterations = 1) 
    # Invert
    # Canny
    #canny = cv2.Canny(clean, 120, 255, 1)
    cv2.imshow("Preprocessed Image", dilated)
    return dilated
