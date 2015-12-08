# -*- coding: utf-8 -*-
import cv2
import sys
import numpy as np

if len(sys.argv) != 2:
    print "Usage: python decapitat.py before.png after.png"
    exit()

# Parameters
imageInPath = sys.argv[1]
imageOutPath = sys.argv[2]

# Haar Cascade
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Read the image
image = cv2.imread(imageInPath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.cv.CV_HAAR_SCALE_IMAGE
)

# Draw a rectangle around the faces
# for (x, y, w, h) in faces:
#    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
# cv2.imshow("Faces", image)
# cv2.waitKey(0)

# Crop image
highestFace = np.argmin(faces[:, 1], 0)
(x, y, w, h) = faces[highestFace, :]
image = image[x: , :]

# Save image
cv2.imwrite(imageOutPath, image)