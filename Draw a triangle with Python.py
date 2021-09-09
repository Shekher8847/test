import numpy as np
import cv2


#we import our libraries. By the help of array() method draw an image of size 512*512 of order 3.  Again using the array() method to create array of the shape type int32.
# remember that points start drawn from the left-topmost position
img = np.zeros((512, 512,3), dtype = "uint8")
rectangle = np.array([[100,50],[100,225],[50,225],[50,50]], np.int32)
triangle = np.array([[120,100],[120,200],[150,100]], np.int32)


rectangleImage =cv2.polylines(img, [rectangle], False, (0,255,0), thickness=3)
triangleImage =cv2.polylines(img, [triangle], False, (0,0,255), thickness=3)
cv2.imshow('Shapes', rectangleImage)
cv2.imshow('Shapes', triangleImage)
cv2.waitKey(0)
cv2.destroyAllWindows()