import cv2
import numpy as np

def segment_characters(plate_image):

    #plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    #_, plate_image = cv2.threshold(plate_image, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(plate_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and sort contours
    char_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 0.1 * plate_image.shape[0] < h < 0.9 * plate_image.shape[0] and w > 5:
            char_contours.append((x, y, w, h))

    char_contours = sorted(char_contours, key=lambda c: c[0])
    
    characters = []
    for x, y, w, h in char_contours:
        char = plate_image[y:y + h, x:x + w]
        characters.append(char)

    return characters