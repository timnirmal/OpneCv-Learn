import cv2
import numpy as np

img = cv2.imread('Materials/flower.jpg')  # Read the image
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to HSV

img_filtered = cv2.inRange(imgHSV, (175, 108, 0), (179, 255, 255))

imgResult = cv2.bitwise_and(img, img, mask=img_filtered)
cv2.imshow("Result", imgResult)

cv2.imshow("Original", img)
cv2.imshow("Result HSV", imgHSV)
cv2.imshow("Img Filtered", img_filtered)

kernel = np.ones((15, 15), np.uint8)
# average = cv2.blur(img_filtered, (15, 15))
averaging_filter = cv2.filter2D(img_filtered, -1, kernel)
averaging = cv2.filter2D(imgResult, -1, kernel)

cv2.imshow("Averaging", averaging)
cv2.imshow("Averaging_filter", averaging_filter)

cv2.waitKey(0)
cv2.destroyAllWindows()
