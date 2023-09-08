import cv2
from skimage import io
import pytesseract
import logging
import os
import pathlib

logging.basicConfig(level=logging.DEBUG)
curr_dir = os.getcwd()
parent_dir = os.path.dirname(curr_dir)
config = ('-l eng --oem 1 --psm 3')

class TextExtractor:
    def __init__(self):
        pass
        
    def extract_text(self, img):
        logging.info("Extracting text...")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        extracted_txt = pytesseract.image_to_string(img_rgb, config=config)
        logging.info(f"image_to_string: {extracted_txt}")
        logging.info("Extraction complete.")