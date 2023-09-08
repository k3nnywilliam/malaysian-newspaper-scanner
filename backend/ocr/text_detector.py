import cv2
import numpy as np
import logging
import pytesseract
from imutils.object_detection import non_max_suppression

class DetectorAndExtractor:
    def __init__(self):
        pass
    
    def detect_text(self, image):
        orig = image.copy()
        (origH, origW) = image.shape[:2]
        (newW, newH) = (args["width"], args["height"])
        rW = origW / float(newW)
        rH = origH / float(newH)
        
        layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]
        
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]
    
        size = (640, 720) 
        blob = cv2.dnn.blobFromImage(image, scalefactor=1/255, size=size, \
            sswapRB=True, crop = False)
        
        net = cv2.dnn.readNet(args["east"])
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        
    def extract_text(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        (boxes, confidence_val) = self.predictions(scores, geometry)
        boxes = non_max_suppression(np.array(boxes), probs=confidence_val)
        
        results = []
        
        (origH, origW) = image.shape[:2]
        (newW, newH) = (args["width"], args["height"])
        
        rW = origW / float(newW)
        rH = origH / float(newH)

        # loop over the bounding boxes to find the coordinate of bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the coordinates based on the respective ratios in order to reflect bounding box on the original image
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            #extract the region of interest
            r = orig[startY:endY, startX:endX]

            #configuration setting to convert image to string.  
            configuration = ("-l eng --oem 1 --psm 8")
            ##This will recognize the text from the image of bounding box
            text = pytesseract.image_to_string(r, config=configuration)
            logging.info(f"image_to_string: {pytesseract.image_to_string(img_rgb)}")

            # append bbox coordinate and associated text to the list of results 
            results.append(((startX, startY, endX, endY), text))
            
    def predictions(prob_score, geo):
        (numR, numC) = prob_score.shape[2:4]
        boxes = []
        confidence_val = []

        # loop over rows
        for y in range(0, numR):
            scoresData = prob_score[0, 0, y]
            x0 = geo[0, 0, y]
            x1 = geo[0, 1, y]
            x2 = geo[0, 2, y]
            x3 = geo[0, 3, y]
            anglesData = geo[0, 4, y]

            # loop over the number of columns
            for i in range(0, numC):
                if scoresData[i] < args["min_confidence"]:
                    continue

                (offX, offY) = (i * 4.0, y * 4.0)

                # extracting the rotation angle for the prediction and computing the sine and cosine
                angle = anglesData[i]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # using the geo volume to get the dimensions of the bounding box
                h = x0[i] + x2[i]
                w = x1[i] + x3[i]

                # compute start and end for the text pred bbox
                endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
                endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
                startX = int(endX - w)
                startY = int(endY - h)

                boxes.append((startX, startY, endX, endY))
                confidence_val.append(scoresData[i])

        # return bounding boxes and associated confidence_val
        return (boxes, confidence_val)