import cv2
import pytesseract
import argparse
import img_resize
import numpy as np

kernel = np.ones((5,5), np.uint8)

def preprocess(image, target=30):
    '''
    Preprocesses the image for Tesseract
    '''
    # correct the text height
    ratio = max(0.5, min(1.5, target / image.shape[0]))
    image = cv2.resize(image,
                       dsize=None,
                       fx=ratio,
                       fy=ratio,
                       interpolation=cv2.INTER_LANCZOS4)

    # apply Otsu thresholding
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # invert image if more than 3/5 of pixels are black
    height, width = thresh.shape
    if (cv2.countNonZero(thresh) / (height * width)) < (2 / 5):
        thresh = cv2.bitwise_not(thresh)

    return thresh

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to input image to be OCR'd")
args = vars(ap.parse_args())

image = cv2.imread(args["image"], cv2.IMREAD_GRAYSCALE)
#image = cv2.resize(image,(1280,720))
image = img_resize.image_resize(image,height=720)

image = preprocess(image)

# use Tesseract to OCR the image
print("Tesseract Loaded!")
text = pytesseract.image_to_string(image)
print(text)
cv2.imshow("Thresholded image", image)
cv2.waitKey(0)