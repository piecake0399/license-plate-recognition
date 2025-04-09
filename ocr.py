import os
import time
import numpy as np
import easyocr
import logging
from logging.handlers import RotatingFileHandler
import cv2
from preprocessing import preprocess_for_ocr

class EasyOcr():
    def __init__(self, lang=['en'], allow_list='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.', min_size=50, log_level='INFO', log_dir='./logs/'):
        self.reader = easyocr.Reader(lang, gpu=False)
        self.allow_list = allow_list
        self.min_size = min_size

        # Logging setup
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

        # Mapping thường nhầm lẫn
        self.dict_char_to_int = {'O': '0', 'I': '1', 'L': '4', 'G': '6', 'S': '5', 'B': '8', 'Z': '2'}
        self.dict_int_to_char = {'0': 'O', '1': 'I', '4': 'L', '6': 'G', '5': 'S', '8': 'B', '2': 'Z'}

    def run(self, detect_result_dict):
        if detect_result_dict['cropped_img'] is not None:
            t0 = time.time()
            img = detect_result_dict['cropped_img']
            img = preprocess_for_ocr(img)
            file_name = detect_result_dict.get('file_name')

            ocr_result = self.reader.readtext(img, allowlist=self.allow_list, min_size=self.min_size)
            text = [x[1] for x in ocr_result]
            confid = [x[2] for x in ocr_result]

            text = "".join(text) if text else None
            text = self.format_license(text) if text else None
            confid = np.round(np.mean(confid), 2) if confid else None
            t1 = time.time()

            is_valid = self.license_complies_format(text) if text else False
            valid_msg = "✔ Biển số hợp lệ" if is_valid else "✘ Biển số không hợp lệ"

            print(f'Recognized number: {text}, conf.:{confid}.\n{valid_msg}\nOCR total time: {(t1 - t0):.3f}s')

            if self.logger:
                self.logger.debug(f'{file_name} Recognized number: {text}, conf.:{confid}, {valid_msg}, OCR total time: {(t1 - t0):.3f}s.')

            return {'text': text, 'confid': confid, 'valid': is_valid}
        else:
            return {'text': None, 'confid': None, 'valid': False}

    def ocr_img_preprocess(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (img_gray.shape[1]*2, img_gray.shape[0]*2), interpolation=cv2.INTER_CUBIC)
        _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_invert = 255 - img_thresh
        return img_invert

    def format_license(self, text):
        if len(text) not in [8, 9]:
            return text

        license_plate_ = ''
        mapping = {
            0: self.dict_char_to_int, 1: self.dict_char_to_int,
            2: self.dict_int_to_char,
            4: self.dict_char_to_int, 5: self.dict_char_to_int, 
            6: self.dict_char_to_int, 7: self.dict_char_to_int, 
            8: self.dict_char_to_int
        }

        for i in range(len(text)):
            if i in mapping and text[i] in mapping[i]:
                license_plate_ += mapping[i][text[i]]
            else:
                license_plate_ += text[i]

        return license_plate_


    def license_complies_format(self, text):
        import string
        if len(text) not in [8, 9]:
            return False

        if not (text[0].isdigit() and text[1].isdigit()):
            return False
        if not (text[2].isalpha() and text[2] in string.ascii_uppercase):
            return False

        # Phần số phía sau
        digits_part = text[3:]
        digits_part = digits_part.replace(".", "")  # nếu có ký tự đặc biệt

        return digits_part.isdigit()
