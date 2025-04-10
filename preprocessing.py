import cv2
import imutils
import math
import numpy as np

def preprocess_for_ocr(image):
    # Resize ảnh để dễ xử lý hơn
    image = imutils.resize(image, width=300)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Làm mượt nhưng vẫn giữ cạnh

    # Phát hiện cạnh
    edged = cv2.Canny(gray, 30, 200)

    # Tìm contour
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    plate_img = None
    imgThresh = None

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            # Tính góc lệch
            (x1, y1) = screenCnt[0][0]
            (x2, y2) = screenCnt[1][0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

            # Tạo mask để crop
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [screenCnt], -1, 255, -1)

            # Crop
            x, y = np.where(mask == 255)
            topx, topy = np.min(x), np.min(y)
            bottomx, bottomy = np.max(x), np.max(y)

            roi_color = image[topx:bottomx, topy:bottomy]
            roi_gray = gray[topx:bottomx, topy:bottomy]

            # Align
            height = bottomx - topx
            width = bottomy - topy
            center = (width / 2.0, height / 2.0)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            plate_img = cv2.warpAffine(roi_color, rot_matrix, (bottomy - topy, bottomx - topx))
            imgThresh = cv2.warpAffine(roi_gray, rot_matrix, (bottomy - topy, bottomx - topx))

            # Phóng to để OCR dễ đọc hơn
            plate_img = cv2.resize(plate_img, (0, 0), fx=3, fy=3)
            imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)
            break

    return imgThresh if imgThresh is not None else gray  # fallback nếu không tìm được biển