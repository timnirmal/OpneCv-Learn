import cv2
import numpy as np

# 1. Wrap Perspective
"""
img = cv2.imread("Materials/cards.jpg")

width, height = 250, 350

pts1 = np.float32([[510, 125], [820, 205], [690, 630], [363, 532]])  # Source image coordinates to be cropped
pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])  # Resulting cropped image coordinates
# By changing the values of pts1 and pts2, the image can be cropped in different ways and perspectives

# Get the perspective transform matrix and apply it to the image
matrix = cv2.getPerspectiveTransform(pts1, pts2)
img_wrapped = cv2.warpPerspective(img, matrix, (width, height))  # Set Result Image Size
# (If exceeded, part of original image will be shown)

# Display the resulting image
cv2.imshow("Original", img)
cv2.imshow("Result", img_wrapped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the resulting image
cv2.imwrite("2.jpg", img)
"""

# 2. Color Filtering
"""
img = cv2.imread('Materials/flower.jpg') # Read the image
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convert to HSV
# HSV is a color space that is more sensitive to color differences
# Hue is the color, Saturation is the intensity, Value is the brightness
# Hue is the angle of the color, Saturation is the distance from the center of the color, Value is the brightness
# H means Hue, S means Saturation, V means Value

img_filtered = cv2.inRange(imgHSV, (0, 0, 0), (180, 255, 255)) # Filter the image
# Filter Red Color
# img_filtered = cv2.inRange(imgHSV, (0, 100, 100), (10, 255, 255))

# Display the resulting image
cv2.imshow("Original", img)
cv2.imshow("Result", imgHSV) # imshow use BGR, not HSV (So not useful for this case)
# View HSV image
cv2.imshow("Result", img_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# 3. Color Filtering with Trackbar
"""
img = cv2.imread('Materials/flower.jpg') # Read the image
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convert to HSV

def empty(i):
    pass


height = 400
width = 500

# Trackbar name, window name, start value, max value, when we change value which fuction to call
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", width, height)

cv2.createTrackbar('Hue Min', 'Trackbars', 0, 179, empty)
cv2.createTrackbar('Hue Max', 'Trackbars', 179, 179, empty)
cv2.createTrackbar('Sat Min', 'Trackbars', 0, 255, empty)
cv2.createTrackbar('Sat Max', 'Trackbars', 255, 255, empty)
cv2.createTrackbar('Val Min', 'Trackbars', 0, 255, empty)
cv2.createTrackbar('Val Max', 'Trackbars', 255, 255, empty)



while True:
    hl = cv2.getTrackbarPos('Hue Min', 'Trackbars') # hue lower
    hu = cv2.getTrackbarPos('Hue Max', 'Trackbars') # hue upper
    sl = cv2.getTrackbarPos('Sat Min', 'Trackbars')
    su = cv2.getTrackbarPos('Sat Max', 'Trackbars')
    vl = cv2.getTrackbarPos('Val Min', 'Trackbars')
    vu = cv2.getTrackbarPos('Val Max', 'Trackbars')

    # Filter the image
    img_filtered = cv2.inRange(imgHSV, (hl, sl, vl), (hu, su, vu)) # Mask

    # Display the resulting image
    cv2.imshow("Original", img)
    cv2.imshow("Result2", img_filtered)

    imgResult = cv2.bitwise_and(img, img, mask=img_filtered)
    cv2.imshow("Result", imgResult)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()


# 74, 179, 33, 255, 20, 255
"""

# 5. Bluing and Smoothing
"""
img = cv2.imread('Materials/flower.jpg') # Read the image
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convert to HSV


img_filtered = cv2.inRange(imgHSV, (175, 108, 0), (179, 255, 255)) # Filter the image
# Filter Red Color

imgResult = cv2.bitwise_and(img, img, mask=img_filtered)
cv2.imshow("Result", imgResult)

# Display the resulting image
cv2.imshow("Original", img)
cv2.imshow("Result HSV", imgHSV) # imshow use BGR, not HSV (So not useful for this case)
# View HSV image
cv2.imshow("Img Filtered", img_filtered)

kernel = np.ones((15, 15), np.float32)/225
#average = cv2.blur(img_filtered, (15, 15))

# Averaging Blur
averaging_filter = cv2.filter2D(img_filtered, -1, kernel)
averaging = cv2.filter2D(imgResult, -1, kernel)
cv2.imshow("Averaging", averaging)
cv2.imshow("Averaging_filter", averaging_filter)

# Gaussian Blur
gaussian_filter = cv2.GaussianBlur(img_filtered, (15, 15), 0)
gaussian = cv2.GaussianBlur(imgResult, (15, 15), 0)
cv2.imshow("Gaussian", gaussian)

# Median Blur
median_filter = cv2.medianBlur(img_filtered, 15)
median = cv2.medianBlur(imgResult, 15)
cv2.imshow("Median", median)

# Bilateral Blur
bilateral_filter = cv2.bilateralFilter(img_filtered, 15, 75, 75)
bilateral = cv2.bilateralFilter(imgResult, 15, 75, 75)
cv2.imshow("Bilateral", bilateral)


cv2.waitKey(0)
cv2.destroyAllWindows()

# Kernel is a matrix that is used to filter the image

# Kernel is a matrix that is used to perform some operation on the image
# Kernel size must be odd
# Kernel size must be positive
# Kernel size must be greater than 1
# Kernel size must be less than image size
# It is like moving a window over the image


"""


# 6. Morphological Operations (Morphological Transformation)

"""
img = cv2.imread('Materials/flower.jpg') # Read the image
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convert to HSV

mask = cv2.inRange(imgHSV, (175, 108, 0), (179, 255, 255)) # Filter Red Color

imgResult = cv2.bitwise_and(img, img, mask=mask)

# Display the resulting image
cv2.imshow("Original", img)
cv2.imshow("Img Filtered", mask)
cv2.imshow("Result", imgResult)

kernel = np.ones((15, 15), np.float32)/225

# Erosion
erosion = cv2.erode(mask, kernel, iterations=4)
erosion_4times = cv2.erode(mask, kernel)
cv2.imshow("Erosion", erosion)
cv2.imshow("Erosion2", erosion_4times)

# Dilation
# Dilation is the opposite of erosion
dilation = cv2.dilate(mask, kernel)
dilation_4times = cv2.dilate(mask, kernel, iterations=4)
cv2.imshow("Dilation", dilation)
cv2.imshow("Dilation2", dilation_4times)

# Opening
# Errosion followed by Dilation (Like a blackhat filter)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
cv2.imshow("Opening", opening)

# Closing
# Dilation followed by Erosion (Like a whitehat filter)
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Closing", closing)

# Gradient
# Difference between dilation and erosion
gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
cv2.imshow("Gradient", gradient)

# Top Hat
# Difference between input image and Opening
top_hat = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernel)
cv2.imshow("Top Hat", top_hat)

# Black Hat
# Difference between closing and input image
black_hat = cv2.morphologyEx(mask, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow("Black Hat", black_hat)


cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# 7. Contours

# contours is a list of all the contours in the image
# Each contour is a list of points (x, y)
# Each contour is a closed shape with an area, a perimeter and a list of points (like borders)

img = cv2.imread('Materials/contour.png') # Read the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale

ret, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV) # Threshold the image

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Find contours
# RETR_TREE: Retrieves all of the contours and reconstructs a full hierarchy of nested contours
# CHAIN_APPROX_SIMPLE: Compresses horizontal, vertical, and diagonal segments and leaves only their end points
# cv2.CHAIN_APPROX_NONE: Retrieves all of the contours without establishing any hierarchical relationships
# cv2.CHAIN_APPROX_SIMPLE: Combines all nearby points on contour boundaries
# cv2.CHAIN_APPROX_TC89_L1: Modifies Teh-Chin chain approximation algorithm by Tom Foehl
# cv2.CHAIN_APPROX_TC89_KCOS: Combines all nearby points on contour boundaries with the Teh-Chin chain approximation algorithm
# cv2.CHAIN_APPROX_TC89_L1: Combines all nearby points on contour boundaries with the Teh-Chin chain approximation algorithm
# cv2.CHAIN_APPROX_TC89_KCOS: Combines all nearby points on contour boundaries with the Teh-Chin chain approximation algorithm

cv2.drawContours(img, contours, -1, (0, 0, 0), 3) # Draw all the contours on the image
# Change -1 to draw a specific contour

# -1: Draw all the contours
# 0, 0, 0: Color of the contours
# 3: Thickness of the contours
print(len(contours)) # Number of contours


cv2.imshow("Original", img)
#cv2.imshow("Gray", gray)
#cv2.imshow("Threshold", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()