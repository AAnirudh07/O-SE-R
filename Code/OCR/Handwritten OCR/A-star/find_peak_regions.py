from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import sobel
import matplotlib.pyplot as plt
import numpy as np

from horizontal_projections import horizontal_projections

def find_peak_regions(hpp, divider=2):
    threshold = (np.max(hpp)-np.min(hpp))/divider
    peaks = []
    peaks_index = []
    for i, hppv in enumerate(hpp):
        if hppv < threshold:
            peaks.append([i, hppv])
    return peaks

if __name__ == "__main__":

    img = rgb2gray(imread("ocr_test_img.jpg"))
    sobel_image = sobel(img)
    hpp = horizontal_projections(sobel_image)

    peaks = find_peak_regions(hpp)
    peaks_index = np.array(peaks)[:,0].astype(int)

    segmented_img = np.copy(img)
    r,c = segmented_img.shape
    for ri in range(r):
        if ri in peaks_index:
            segmented_img[ri, :] = 0
            
    plt.figure(figsize=(20,20))
    plt.imshow(segmented_img, cmap="gray")
    plt.show()