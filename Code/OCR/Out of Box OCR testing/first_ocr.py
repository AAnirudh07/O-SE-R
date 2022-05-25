import cv2
import pytesseract
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to input image to be OCR'd")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# use Tesseract to OCR the image
print("Tesseract Loaded!")
text = pytesseract.image_to_string(image)
print(text)