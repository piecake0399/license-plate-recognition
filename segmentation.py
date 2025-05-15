import cv2
import numpy as np
from char_classification import predict_character

def segment_characters(plate_image):

    #plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    #_, plate_image = cv2.threshold(plate_image, 127, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(plate_image, 120, 255, 1)

    # Find contours
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and sort contours
    char_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        ar = w / float(h)
        if 0.2 < ar < 1.0 and 100 < w*h:
            char_contours.append((x, y, w, h))

    char_contours = sorted(char_contours, key=lambda b: (b[0], b[1]))

    # Draw bounding boxes around characters
    for (x, y, w, h) in char_contours:
        cv2.rectangle(plate_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("Contours", plate_image)
    
    # Predict
    plate_str = ""
    for x, y, w, h in char_contours:
        roi = plate_image[y : y + h, x : x + w]
        ch = predict_character(roi)
        plate_str += str(ch)

    return plate_str