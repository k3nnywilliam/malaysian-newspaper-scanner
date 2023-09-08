import cv2
from skimage import io
import pytesseract
import logging
import os
import pathlib

logging.basicConfig(level=logging.DEBUG)
logging.debug('This will get logged')
curr_dir = os.getcwd()
parent_dir = os.path.dirname(curr_dir)

class Scanner:
    def __init__(self):
        self.img = None
        self.width = int()
        self.height = int()
        self.scaled_height = int()
        self.scaled_width = int()
        
    def read_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(pytesseract.image_to_string(img_rgb))
        
    def show_image(self, img):
         logging.info("Rendering the image...")
         cv2.imshow("resized", img)
         cv2.waitKey(0)
         cv2.destroyAllWindows()
         logging.info("Done.")
        
    
#img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
#print(pytesseract.image_to_string(img_rgb))

