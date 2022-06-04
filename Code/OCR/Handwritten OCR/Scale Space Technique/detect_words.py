from word_detector import prepare_img, detect, sort_line
import matplotlib.pyplot as plt
import cv2

# (1) prepare image:
# (1a) convert to grayscale
# (1b) scale to specified height because algorithm is not scale-invariant
img = prepare_img(cv2.imread('../ocr_test_img.jpg'), 50)

# (2) detect words in image
detections = detect(img,
                    kernel_size=25,
                    sigma=11,
                    theta=7,
                    min_area=100)

# (3) sort words in line
line = sort_line(detections)[0]

#print(line)

# (4) show word images
plt.subplot(len(line), 1, 1)
plt.imshow(img, cmap='gray')
print(len(line))
for i, word in enumerate(line):
  print(word.bbox)
  print(word.img)
  #plt.subplot(len(line), 1, i + 2)
  #plt.imshow(word.img, cmap='gray')
plt.show()
