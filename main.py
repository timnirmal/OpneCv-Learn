# In OpenCv its BGR not RGB

import cv2
import numpy as np

# 1. Showing Image
""""


img = cv2.imread('Materials/car.jpg')
# Read image black and white
img_gray = cv2.imread('Materials/car.jpg', 0)

img[0:100, 0:100] = 0, 0, 0

cv2.imshow('image', img)
cv2.imshow('image_gray', img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print image size
print(img.shape)
print(img_gray.shape)

"""

# 2. Using Webcam

"""

cap = cv2.VideoCapture(0)   # Change value to select camera

while cap.isOpened():
    ret, frame = cap.read()
    # ret is True if frame is read correctly (return value)
    # frame is the image matrix

    if ret:
        cv2.imshow('Webcam', frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Webcam_gray', gray)
    else:
        break

    # waitKey(0) is wait for infinite time until any key is pressed
    # waitKey(1) is for 1 ms (Wait for 1 ms)
    # 30 is for 30 fps (30 frames per second, ideal gap)
    # If use didnt press q in 1ms, -1 is returned, and video will continue
    # If user pressed any key, we can use cv2.waitKey(1) > -1 or cv2.waitKey(1) != -1 
    if cv2.waitKey(1) & 0xFF == ord('q'):   # or if cv2.waitKey(1) == 27: # (ESC key)
        break

cap.release()
cv2.destroyAllWindows()

"""

# 3. Using Video
"""


# Same code for Webcam, IP address of IP camera, Video File
cap = cv2.VideoCapture('Materials/lane.mkv')

while cap.isOpened():

    ret, frame = cap.read()
    if ret:
        cv2.imshow('Video', frame)
        print('Video is playing')
    else:
        break

    # since waitKey(1) is 1 ms, video will play fast. If you want to slow it down, use waitKey(30)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""

# 4. Change Video Resolution, Size, FPS and Codec
"""

cap = cv2.VideoCapture('Materials/lane.mkv')
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # 3 is for width, cv2.CAP_PROP_FRAME_WIDTH
# cap.set(4, 206) # 4 is for height, cv2.CAP_PROP_FRAME_HEIGHT
# cap.set(5, 30)  # 5 is for fps, cv2.CAP_PROP_FPS
# cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
# cap.set(cv2.CAP_PROP_CONTRAST, 0)
# cap.set(cv2.CAP_PROP_SATURATION, 0)
# cap.set(cv2.CAP_PROP_GAIN, 0)
# cap.set(cv2.CAP_PROP_FPS, 30)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 3
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) # 4


while cap.isOpened():
    ret, frame = cap.read()
    print('ret', ret)
    print('frame', frame)  # Frame is set of BGR values ex : [18, 25, 120]

    if ret:
        cv2.imshow('Video', frame)
        resized = cv2.resize(frame, (340, 240))
        cv2.imshow('Video_resized', resized)
    else:
        break

    # since waitKey(1) is 1 ms, video will play fast. If you want to slow it down, use waitKey(30)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""

# 5. Draw shapes and Text
"""

# plane canvas 640x640
img = np.zeros((640, 640, 3), np.uint8)
# narray to opencv image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Draw line
cv2.line(img, (0, 0), (640, 640), (255, 0, 0), 5)
cv2.line(img, (0, 640), (640, 0), (0, 255, 0), 5)

# draw rectangle
cv2.rectangle(img, (0, 0), (200, 200), (0, 255, 0), 5)

# draw circle
cv2.circle(img, (320, 240), 100, (0, 0, 255), 5)

# draw filled rectangle
cv2.rectangle(img, (320, 240), (420, 340), (0, 0, 255), -1)

# draw ellipse
cv2.ellipse(img, (320, 240), (100, 50), 0, 0, 360, (255, 0, 0), 5)

# draw polygon
pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(img, [pts], True, (0, 220, 220),3)

# draw filled polygon
contours = np.array([[100, 50], [200, 300], [70, 200], [500, 100]], np.int32)
# move contours by 0, -200
cv2.fillPoly(img, [contours], (0, 255, 255))

# draw text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)

# show
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""

# 6. Image Addition
"""
img1 = cv2.imread('Materials/img1.jpg')
img2 = cv2.imread('Materials/img2.jpg')

add = cv2.add(img1, img2)
add_row = img1+ img2
add_weighted = cv2.addWeighted(img1, 0.7, img2, 0.3, 0.5)

cv2.imshow("add", add)
#cv2.imshow("row addtion", add[0:add_row, :])
cv2.imshow("row addtion", add_row)
cv2.imshow("weighted ", add_weighted)

cv2.waitKey(0)
cv2.destroyAllWindows()


# explanation of what happened
x = np.int8([250])
y = np.nit8([10])

print(x+y)    # Since 260 > 255, 260 % 256 is taken
# So answer is 4
print(cv2.add(x , y))  # Since 260 > 255, This will save maximum values 255
# so answer is 255
"""

# 7. Threshold

img = cv2.imread("Materials/bookpage.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, threshold = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
# if f(x,y) < T:
#   then f(x,y) = 0
# else:
#   f (x,y) = 255

cv2.imshow("Original Image", img)
cv2.imshow("Gray version", gray)
cv2.imshow("Threshold Version 10", threshold)

cv2.waitKey(0)
cv2.destroyAllWindows()


