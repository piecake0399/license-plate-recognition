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
    plate_h, plate_w = plate_image.shape[:2]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        char_area = w * h
        plate_area = plate_w * plate_h
        area_ratio = char_area / float(plate_area)
        aspect_ratio = w / float(h)
        if (0.01 < area_ratio < 0.2) and (0.2 < aspect_ratio < 0.6):
            char_contours.append((x, y, w, h))

    char_contours = sorted(char_contours, key=lambda b: (b[1], b[0]))
    canny_bgr = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)    
    
    # Predict
    plate_str = ""
    for x, y, w, h in char_contours:
        roi = plate_image[y : y + h, x : x + w]
        ch = predict_character(roi)
        plate_str += str(ch)
        cv2.putText(canny_bgr, ch, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw bounding boxes around characters
    for (x, y, w, h) in char_contours:
        cv2.rectangle(canny_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Contours", canny_bgr)

    return plate_str