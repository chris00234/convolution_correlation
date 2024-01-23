import numpy as np
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


fig = plt.figure()

img = mpimg.imread('./conv.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(1,2,1)
plt.imshow(img)


# img = cv2.imread("/Users/chris/Documents_local/computer_vision_116/Assignment_written/conv.jpeg")
# # print(img.shape)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(img.shape)
# cv2.imshow('img', img)
# cv2.waitKey(0)
I = np.array(img)
# print(img)
# # print(I.shape)

kernel = np.array([[1,0,0], [0,0,0], [0,0,0]])

conv = ndimage.correlate(img, kernel, mode = 'constant', cval=0.0)
plt.subplot(1,2,2)
plt.imshow(conv)
# cv2.waitKey(0)
plt.show()


cv2.bitwise_and()