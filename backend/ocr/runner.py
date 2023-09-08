from image_preprocessor import ImagePreprocessor
from text_extractor import TextExtractor
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Please provide the image", type=str, required=True)
    #parser.add_argument("-s", "--save", help="Please provide the image", type=str, required=True)
    args = parser.parse_args()
    processor = ImagePreprocessor()
    width = 1134
    height = 2016
    img = processor.read_image(args.file, width, height)
    gray = processor.convert_to_grayscale(img)
    #processor.write_image(gray, args.save, 600, 1000)
    #processor.show_image(gray)
    warped = processor.edge_extraction(gray, img, width, height)
    #processor.show_contour_image(img, contoured)
    #processor.show_image(img)
    denoised = processor.denoise_image(warped)
    #processor.show_warped(warped)
    #binarized =  processor.dilate_erode(warped)
    #processor.show_image(binarized)
    extractor = TextExtractor()
    extractor.extract_text(warped)