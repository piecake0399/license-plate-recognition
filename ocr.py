import os
import time
import numpy as np
import easyocr
import logging
from logging.handlers import RotatingFileHandler
from scipy.ndimage import rotate
import cv2

class EasyOcr():
    def __init__(self, lang=['en'], allow_list='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.', min_size=50, log_level='INFO', log_dir='./logs/'):
        self.reader = easyocr.Reader(lang, gpu=False)
        self.allow_list = allow_list
        self.min_size = min_size
        self.log_level = log_level
        if self.log_level:
            self.num_log_level = getattr(logging, log_level.upper(), 20)
            self.log_dir = log_dir
            os.makedirs(self.log_dir, exist_ok=True)
            log_formatter = logging.Formatter("%(asctime)s %(message)s")
            logFile = os.path.join(self.log_dir, 'ocr.log')
            my_handler = RotatingFileHandler(logFile, mode='a', maxBytes=25 * 1024 * 1024,
                                             backupCount=10, encoding='utf-8', delay=False)
            my_handler.setFormatter(log_formatter)
            my_handler.setLevel(self.num_log_level)
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(self.num_log_level)
            self.logger.addHandler(my_handler)

        # Ký tự hay nhầm lẫn
        self.dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
        self.dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

    def run(self, detect_result_dict):
        if detect_result_dict['cropped_img'] is not None:
            t0 = time.time()
            img = detect_result_dict['cropped_img']
            img = self.ocr_img_preprocess(img)
            file_name = detect_result_dict.get('file_name')
            ocr_result = self.reader.readtext(img, allowlist=self.allow_list, min_size=self.min_size)
            text = [x[1] for x in ocr_result]
            confid = [x[2] for x in ocr_result]
            text = "".join(text) if len(text) > 0 else None
            text = self.format_license(text) if text else None
            confid = np.round(np.mean(confid), 2) if len(confid) > 0 else None
            t1 = time.time()
            print(f'Recognized number: {text}, conf.:{confid}.\nOCR total time: {(t1 - t0):.3f}s')
            if self.log_level:
                self.logger.debug(f'{file_name} Recognized number: {text}, conf.:{confid}, OCR total time: {(t1 - t0):.3f}s.')

            return {'text': text, 'confid': confid}
        else:
            return {'text': None, 'confid': None}

    def ocr_img_preprocess(self, img):
        # Grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize
        img_gray = cv2.resize(img_gray, (img_gray.shape[1]*2, img_gray.shape[0]*2), interpolation=cv2.INTER_CUBIC)

        # Threshold
        _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert
        img_invert = 255 - img_thresh

        return img_invert

    def format_license(self, text):
        """
        Format lại biển số dựa vào vị trí và ký tự dễ nhầm.
        Vị trí 0-1-4-5-6 thường là chữ → map số thành chữ.
        Vị trí 2-3 thường là số → map chữ thành số.
        """
        if len(text) != 7:
            return text

        license_plate_ = ''
        mapping = {
            0: self.dict_int_to_char, 1: self.dict_int_to_char,
            2: self.dict_char_to_int, 3: self.dict_char_to_int,
            4: self.dict_int_to_char, 5: self.dict_int_to_char, 6: self.dict_int_to_char
        }

        for i in range(7):
            char = text[i]
            if char in mapping[i]:
                license_plate_ += mapping[i][char]
            else:
                license_plate_ += char

        return license_plate_
