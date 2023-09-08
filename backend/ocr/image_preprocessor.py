from distutils.file_util import write_file
from typing import final
import cv2
import logging
import os
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logging.debug('This will get logged')
curr_dir = os.getcwd()
parent_dir = os.path.dirname(curr_dir)

class ImagePreprocessor:
    def __init__(self):
        self.width = int()
        self.height = int()
        self.scaled_height = int()
        self.scaled_width = int()
    
    def read_image(self, file_path, scale_w, scale_h):
        try:
            logging.info(f"Current path: {curr_dir}/{file_path}")
            logging.info(f"Parent Directory:{parent_dir}")
            logging.info("Reading the image...")
            img = cv2.imread(f"{curr_dir}/{file_path}")
            rescaled_w = scale_w #1134
            rescaled_h = scale_h #2016
            width = int(img.shape[1])
            height = int(img.shape[0])
            resized_img = cv2.resize(img, (rescaled_w, rescaled_h),  interpolation = cv2.INTER_AREA)
            logging.info(f"Original dim: {width},{height}")
            logging.info(f"Modified dim: {width},{height}")
            return resized_img
        except IOError:
            logging.error("IO Error: Check image path.")
            logging.info(f"Image path: {curr_dir}{file_path}")
            
    def convert_to_grayscale(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray 
    
    def set_scale(self, scale = 50):
        self.scaled_width = int(self.width * scale / 100)
        self.scaled_height = int(self.height * scale / 100)
        logging.info(f"Scaled width: {self.scaled_width}")
        logging.info(f"Scaled height: {self.scaled_height}")
        
    def write_image(self, img, myfile, width = 0, height = 0):
        logging.info("Saving...")
        curr_dir = os.getcwd()
        logging.info(f"list dir: {os.listdir(curr_dir)}")
        resized = cv2.resize(img, (width, height))
        cv2.imwrite('mypic.jpg', resized)
        logging.info("saved...")
        #return resized
    
    def order_points(self, pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype = "float32")
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        # return the ordered coordinates
        return rect
    
    def four_point_transform(self, image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped
    
    def denoise_image(self, img):
        noiseless = cv2.fastNlMeansDenoising(img, None, 20, 7, 21)
        return noiseless
    
    def edge_extraction(self,gray, orig_img, width, height):
        #denoised = self.denoise_image(gray)
        #blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in contours:
            # approximate each contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
            # check if we have found our document
            if len(approx) == 4:
                doc_cnts = approx
                break
        warped = self.four_point_transform(orig_img, doc_cnts.reshape(4, 2))
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        final_img = cv2.resize(warped, (width, height))
        return final_img
        
    def show_image(self, img):
        try:
            logging.info("Showing the image...")
            cv2.imshow("resized", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            logging.info("Windows closed.")
        except IOError:
            logging.error("Can't process the image")
            
    def show_warped(self, warped_img):
        try:
            logging.info("Showing the image...")
            cv2.imshow("Warped", warped_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except IOError:
            logging.error("Can't show any image.")
        finally:
            logging.info("Show warped done.")
    
    def dilate_erode(self, img):
        blur = cv2.GaussianBlur(img,(5,5),0)
        #ret3, th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        #     cv2.THRESH_BINARY,11,2)
        return blur
            
    def show_contour_image(self, img, contours):
        try:
            cv2.imshow("Image", img)
            cv2.drawContours(img, contours, -1, (0,256,0), 3)
            cv2.imshow("All contours", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except IOError:
            logging.error("Cannot show contour image")
        finally:
            logging.info("Show contour done.")



