from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import sobel
import matplotlib.pyplot as plt
import numpy as np

def horizontal_projections(sobel_image):
    #threshold the image.
    sum_of_rows = []
    for row in range(sobel_image.shape[0]-1):
        sum_of_rows.append(np.sum(sobel_image[row,:]))
    
    return sum_of_rows

if __name__ == "__main__":
    img = rgb2gray(imread("ocr_test_img.jpg"))
    sobel_image = sobel(img)
    hpp = horizontal_projections(sobel_image)

    plt.figure(figsize=(10,10))
    plt.axis("off")
    plt.plot(hpp)
    plt.show()
