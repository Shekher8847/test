import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imutils
import os
yolo = cv2.dnn.readNet("C:\\Users\cd42146\Downloads\yolov3-tiny.weights","C:\\Users\cd42146\Downloads\yolov3-tiny.cfg")

classes =[]

with open("C:\\Users\cd42146\Downloads\coco.txt",'r') as f:
    classes= f.read().splitlines()

# print(len(classes))
# print(classes)

img =cv2.imread("C:\\Users\cd42146\Downloads\DSC_0178.JPG")
blob = cv2.dnn.blobFromImage(img,1/255,(320,320),(0,0,0),swapRB=True, crop=False)
print(blob.shape)

# to print image

i = blob[0].reshape(320,320,3)
print(plt.imshow(i))

# suppose we have to set above said blob image as input image

yolo.setInput(blob)

output_layer_name =yolo.getUnconnectedOutLayersNames()

layeroutput = yolo.forward(output_layer_name)

boxes = []
confidence  = []
class_ids = []

for output in layeroutput:
    for detection in output:
        score  =detection[5:]
        class_ids =np.argmax(score)
        confidence = score[class_ids]

        if confidence > 0.7:

            center_x = int(detection[0]*width)
            center_y = int(detection[0] * height)
            w = int(detection[0] * width)
            h = int(detection[0] * height)



            x = int(center_x -w/2)
            y = int(center_y -h/2)

            boxes.append([x,y,w,h])
            confidence.append(float(confidence))
            class_ids.append(class_ids)



len(boxes)

indexes =cv2.dnn.NMSBoxes(boxes,confidence,0.5,0.4)

font = cv2.FONT_HERSHEY_PLAIN
colors =np.random.uniform(0,255, size = ( len(boxes),3))

for i in indexes.flatten():
    x,y,w,h =boxes[i]

    label = str(classes[class_ids[i]])
    confi = str(round(confidence[i],2))
    color = color[i]

    cv2.rectangle(img,(x,y),(x+w,y+h),color,3)
    cv2.putText(img,label + " "+confi,(x ,y+20),font,2,(255,255,255),1)

plt.imshow(img)


# cv2.imwrite(" ",img)









