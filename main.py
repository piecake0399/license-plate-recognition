import os
import glob
import cv2
import argparse
from ultralytics import YOLO
from preprocess import preprocess
from segmentation import segment_characters
from char_classification import predict_character

# Load Model
model = YOLO('best2.pt')

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True, help='Path to image/video/folder or webcam (usb0)')
args = parser.parse_args()
source = args.source

img_source = args.source

# Determine input type
if source.lower().startswith('usb'):
    input_type = 'webcam'
    cap = cv2.VideoCapture(int(source[3:]))
elif os.path.isfile(source):
    _, ext = os.path.splitext(source)
    if ext.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        input_type = 'video'
        cap = cv2.VideoCapture(source)
    else:
        input_type = 'image'
        image_paths = [source]
elif os.path.isdir(source):
    input_type = 'folder'
    image_paths = glob.glob(os.path.join(source, '*'))
else:
    raise ValueError(f"Input is not detected from {source}")

# Main loop
def process_frame(frame):
    results = model(frame)[0]
    detections = results.boxes

    for box in detections:
        xyxy = box.xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        conf = box.conf.item()

        if conf > 0.4:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cropped = frame[ymin:ymax, xmin:xmax]
            if cropped.size > 0:
                processed = preprocess(cropped)
                detect_result_dict = {
                    'cropped_img': cropped,
                    'file_name': 'frame'
                }

                predicted = segment_characters(processed)
                print(f"Predicted string: {predicted}")
                cv2.putText(frame, predicted, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                #cv2.imshow(f'Detected Object', processed)
    return frame

# Handle input types 
if input_type in ['image', 'folder']:
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Không đọc được ảnh {img_path}, bỏ qua.")
            continue
        frame = process_frame(frame)
        cv2.imshow("License Plate Detection", frame)
        key = cv2.waitKey(0)
        if key == ord('q') or key == ord('Q'):
            break

elif input_type in ['video', 'webcam']:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame)
        cv2.imshow("License Plate Detection", frame)
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            break
    cap.release()

cv2.destroyAllWindows()