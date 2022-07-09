import numpy as np
import matplotlib.pyplot as plt
import cv2 as opencv

#frame = opencv.imread("FarFarAway.jpg")
frame = opencv.imread("FarAway.jpg")
print frame.shape
plt.figure(1)
plt.imshow(frame)

plt.show()

