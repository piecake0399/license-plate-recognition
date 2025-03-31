import os
import glob
import cv2
import easyocr
from ultralytics import YOLO
from ocr import EasyOcr
from preprocessing import preprocess_for_ocr

model = YOLO('best2.pt')  #Model goes here
ocr_model = EasyOcr(lang=['en'], allow_list='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', min_size=50, log_level='INFO')

#Open webcam
#cap = cv2.VideoCapture(0)

image_folder = r"C:\Users\kient\Downloads\Projects\YOLO\Nhan dien bien so xe may\test" #Change dir here
image_paths = glob.glob(os.path.join(image_folder, '*'))

#Main loop
for img_path in image_paths:
    frame = cv2.imread(img_path)

    if frame is None:
        print(f"Không đọc được ảnh {img_path}, bỏ qua.")
        continue

    #YOLO detection
    results = model(frame)[0]
    detections = results.boxes

    #Create Bbox
    for box in detections:
        xyxy = box.xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        conf = box.conf.item()
        
        if conf > 0.5:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            #Crop
            cropped = frame[ymin:ymax, xmin:xmax]
            preprossed = preprocess_for_ocr(cropped)
            if cropped.size > 0:
                cv2.imshow('Detected Object', preprossed)
                detect_result_dict = {
                    'cropped_img': cropped,
                    'file_name': 'webcam_frame'
                }

                ocr_result = ocr_model.run(detect_result_dict)
                plate_text = ocr_result['text'] if ocr_result['text'] else "N/A"

                cv2.putText(frame, plate_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("YOLO Detection", frame)

    key = cv2.waitKey(0)
    if key == ord('q') or key == ord('Q'):
        break

#cap.release()
cv2.destroyAllWindows()