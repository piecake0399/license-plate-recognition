import os
import time
import numpy as np
import easyocr
import logging
from logging.handlers import RotatingFileHandler
import cv2
from preprocessing import preprocess_for_ocr

class EasyOcr():
    # --- Initialize OCR ---
    def __init__(self, lang=['en'], allow_list='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.', min_size=50, log_level='INFO', log_dir='./logs/'):
        self.reader = easyocr.Reader(lang, gpu=True)
        self.allow_list = allow_list
        self.min_size = min_size

        # --- Logging setup ---
        self.logger = logging.getLogger(__name__)
        if log_level:
            self.num_log_level = getattr(logging, log_level.upper(), 20)
            self.log_dir = log_dir
            os.makedirs(self.log_dir, exist_ok=True)
            log_formatter = logging.Formatter("%(asctime)s %(message)s")
            logFile = os.path.join(self.log_dir, 'ocr.log')
            my_handler = RotatingFileHandler(logFile, mode='a', maxBytes=25 * 1024 * 1024,
                                             backupCount=10, encoding='utf-8', delay=False)
            my_handler.setFormatter(log_formatter)
            my_handler.setLevel(self.num_log_level)
            self.logger.setLevel(self.num_log_level)
            self.logger.addHandler(my_handler)
        else:
            logging.basicConfig(level=logging.INFO)

    # --- Run ---
    def run(self, detect_result_dict):
        if detect_result_dict['cropped_img'] is not None:
            t0 = time.time()
            img = detect_result_dict['cropped_img']
            img = preprocess_for_ocr(img) # Preprocessing
            file_name = detect_result_dict.get('file_name')

            ocr_result = self.reader.readtext(img, allowlist=self.allow_list, min_size=self.min_size)
            text = [x[1] for x in ocr_result]
            confid = [x[2] for x in ocr_result]

            # --- Text joining ---
            text = "".join(text) if text else None
            confid = np.round(np.mean(confid), 2) if confid else None
            t1 = time.time()

            print(f'Recognized number: {text}, conf.:{confid}.\nOCR total time: {(t1 - t0):.3f}s')

            if self.logger:
                self.logger.debug(f'{file_name} Recognized number: {text}, conf.:{confid}, OCR total time: {(t1 - t0):.3f}s.')

            return {'text': text, 'confid': confid}
        else:
            return {'text': None, 'confid': None} # Return to none if no license plate is detected
