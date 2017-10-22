# import the necessary packages
import numpy as np
import cv2

imageOriginal = cv2.imread("test_original.jpg")

# load the image, convert it to grayscale, and blur it
image = cv2.imread("test_markedup1.jpg")

# remove all non red color
# Convert BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# define range of red color in HSV
lower_blue = np.array([0, 100, 100])
upper_blue = np.array([10, 255, 255])

# Threshold the HSV image to get only red colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(image, image, mask=mask)

image = res

image = cv2.GaussianBlur(image, (3, 3), 0)
edged = cv2.Canny(image, 10, 250)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

# find contours (i.e. the 'outlines') in the image and initialize the
# total number of books found
(_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total = 0

# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # if the approximated contour has four points, then assume that the
    # contour is a book -- a book is a rectangle and thus has four vertices
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        crop_img = imageOriginal[y:(y+h), x:(x+w)]
        cv2.imwrite('rectangle' + str(total) + '.jpg', crop_img)
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)
        total += 1

# display the output
cv2.imwrite('rectangledetection.png', image)
cv2.waitKey(0)
