# Imports
import swtloc as swt
import numpy as np
# Image Path
imgpath = '../ocr_test_img_3.jpg'
# Result Path
respath = 'D:/Projects/O-SE-R/Code/OCR/Handwritten OCR/Segmentation+Extraction'
# Initializing the SWTLocalizer class with the image path
swtl = swt.SWTLocalizer(image_paths=imgpath)
swtImgObj = swtl.swtimages[0]
# Perform Stroke Width Transformation
swt_mat = swtImgObj.transformImage(maximum_angle_deviation=np.pi/2,
                                   gaussian_blurr_kernel=(9, 9),
                                   minimum_stroke_width=3,
                                   maximum_stroke_width=50,
                                   include_edges_in_swt=False,
                                   display=False)  # NOTE: Set display=True 

# Localizing Letters
localized_letters = swtImgObj.localizeLetters(minimum_pixels_per_cc=400,
                                              maximum_pixels_per_cc=5000,
                                              display=False)  # NOTE: Set display=True 

# Calculate and Draw Words Annotations
localized_words = swtImgObj.localizeWords(display=True)  # NOTE: Set display=True 
word_labels = [int(k) for k in list(localized_words.keys())]