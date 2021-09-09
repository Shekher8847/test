# import cv2
# import numpy as np
#
# img = cv2.imread('C:\\Users\cd42146\Downloads\Tractor with SMV.jpg',1)
# # how to draw a line on the pic
# img =  cv2.line(img,(0,0),(255,255),(147,96,4),5)
# img = cv2.arrowedLine(img,(0,255),(255,255),(255,0,255),5)
#
# img = cv2.rectangle(img,(285,300),(475,128),(0,0,255),5)
#
# img = cv2.circle(img,(447,300),65,(0,0,255),5)
#
# font = cv2.FONT_HERSHEY_SIMPLEX
# img= cv2.putText(img,'open cv',(10,400),font,4,(0,255,255),10,cv2.LINE_AA)
#
#
# cv2.imshow('image',img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#---------------------------------------------------------------------------------------------------------
#
# import cv2
# # import time
# import datetime
#
# # capWebcam = cv2.VideoCapture(0)
# # time.sleep(1.000)
#
# cap  = cv2.VideoCapture(0)
# print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# cap.set(3,700)
# cap.set(4,700)
#
# print(cap.get(3))
# print(cap.get(4))
#
# while(cap.isOpened()):
#     ret,frame = cap.read()
#
#     if ret == True:
#
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         text = 'Width: '+ str(cap.get(3))+ 'Height' + str(cap.get(4))
#
#         datet = str(datetime.datetime.now())
#         cv2.putText(frame,datet,(10,50),font,1,(0,255,255),2,cv2.LINE_AA)
#
#         # gray = cv2.cvtColor(frame,cv2.COLOR_BAYER_BG2GRAY,dst=None,dstCn=None)
#         cv2.imshow('frame',frame)
#     if frame is not None:
#         gray = cv2.cvtColor(frame,cv2.COLOR_BAYER_BG2GRAY)
#     # else:
#     #     print('empty frame')
#     #     exit(1)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#         else:
#             break
# cap.release()
# cv2.destroyAllWindows()

#----------------------------------------------------------------------------------------------

# # to see all the events in CV2
# import cv2
# event = {i for i in dir(cv2) if 'EVENT' in i}
# print(event)

#----------------------------------------------------------------------------------------------
#Writing the cordinates on picture

# import cv2
# import numpy as np
# def click_event(event,x,y,flags,param):
#     if event ==cv2.EVENT_LBUTTONDOWN:
#         print(x,',',y)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         strxy = str(x) + ',' + str(y)
#         cv2.putText(img,strxy,(x,y),font,0.5,(255,255,0),2)
#         cv2.imshow('image',img)
#     if event ==cv2.EVENT_RBUTTONDOWN:
#         blue = img[y,x,0]
#         green = img[y, x, 1]
#         red = img[y, x, 2]
#
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         strbgr = str(blue) + ',' + str(green) + ',' +str(red)
#         cv2.putText(img,strbgr,(x,y),font,0.5,(255,0,255),2)
#         cv2.imshow('image',img)
#
#
# # img= np.zeros((512,512,3),np.uint8)
# img = cv2.imread('C:\\Users\cd42146\Downloads\Tractor with SMV.jpg',1)
# cv2.imshow('image',img)
#
# cv2.setMouseCallback('image',click_event)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#------------------------------------------------------------------------------------------------------
# Drawing line between two points ..This is amazing to use on setelite images
# import cv2
# import numpy as np
# def click_event(event,x,y,flags,param):
#     if event ==cv2.EVENT_LBUTTONDOWN:
#         cv2.circle(img,(x,y),3,(0,0,255),-1)
#         points.append((x,y))
#         if len(points)>=2:
#             cv2.line(img,points[-1],points[-2],(255,0,0),5)
#         cv2.imshow('image',img)
#
#
#
# img = np.zeros((512,512,3),np.uint8)
#
# #img = cv2.imread('C:\\Users\cd42146\Downloads\Tractor with SMV.jpg',1)
# cv2.imshow('image',img)
# #
# points =[]
# cv2.setMouseCallback('image',click_event)
# #
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#------------------------------------------------------------------------

# if you click on the pic at a point , new window will be open of same color
#  where pointer were placed
# import cv2
# import numpy as np
#
# def click_event(event,x,y,flags,param):
#     if event ==cv2.EVENT_LBUTTONDOWN:
#         blue = img[x,y,0]
#         green = img[x, y, 1]
#         red  = img[x, y, 2]
#         cv2.circle(img,(x,y),3,(0,0,255),-1)
#
#         mycolorimage =np.zeros((512,512,3),np.uint8)
#
#         mycolorimage[:] = [blue,green,red] # to fill entire image
#
#         cv2.imshow('color',mycolorimage)
#
#
#
# img = np.zeros((512,512,3),np.uint8)
#
# # img = cv2.imread('C:\\Users\cd42146\Downloads\Tractor with SMV.jpg',1)
# cv2.imshow('image',img)
# #
# points =[]
# cv2.setMouseCallback('image',click_event)
# #
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#________________________________________________________________________________________________________________


# some of the arithematic options

# import cv2
# import numpy as np
#
# img = cv2.imread('C:\\Users\cd42146\Downloads\Tractor with SMV.jpg',1)
# img2 = cv2.imread('C:\\Users\cd42146\Downloads\output.JPG',1)
# print(img2.shape)
#
# # print(img.shape) # to return a tuple of number of row ,column and channles (559,758,3)
# # print(img.size)# return the total number of pixel
# # print(img.dtype)# return data type obtained
# # # already find the cordinates of object of interest
# # ball = img[200:180 ,130:200]
# # img[173:133,100:100] = ball
#
#
# # adding two images  .... Amazing feature of CV2
# img = cv2.resize(img,(512,512))
# img2= cv2.resize(img2,(512,512))
# dst = cv2.add(img,img2)
# # merging with weighted method
# dst = cv2.addWeighted(img,.5,img2,.5,0)

#
# cv2.imshow('image',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#----------------------------------------------------------------------------------------------

# bitwise operation , work on mask- pixle wise work
#
# import cv2
# import numpy as np
#
# img1 = np.zeros((512,512,3),np.uint8)
# img1 = cv2.rectangle(img1,(200,0),(300,100),(255,255,255),-1)
# img2 = cv2.imread('C:\\Users\cd42146\Downloads\output.JPG',1)
#
# cv2.imshow("img1",img1)
# cv2.imshow("img2",img2)
#
# img1 = cv2.resize(img1,(512,512))
# img2= cv2.resize(img2,(512,512))
# #
# # bitAnd = cv2.bitwise_and(img2,img1)
# # cv2.imshow('bitAnd',bitAnd)
#
# bitOr = cv2.bitwise_or(img2,img1)
# cv2.imshow('bitOr',bitOr)
#
# bitXOr = cv2.bitwise_xor(img2,img1)
# cv2.imshow('bitXOr',bitXOr)
#
# bitNot = cv2.bitwise_not(img2,img1)
# cv2.imshow('bitNot',bitNot)
#
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#------------------------------------------------------------------------------------------

# Track Bar how you want to change values dynemically in pixcel , you can change
# R G B values with sliders of individual

# import cv2
# import numpy as np
#
# def nothing(x):
#     print(x)
#
# img =np.zeros((300,512,3),np.uint8)
#
# # img = cv2.imread('C:\\Users\cd42146\Downloads\output.JPG',1)
# cv2.namedWindow('image')
#
# cv2.createTrackbar('B','image',0,255,nothing ) # to creat slide bar
# cv2.createTrackbar('G','image',0,255,nothing )
# cv2.createTrackbar('R','image',0,255,nothing )
#
# switch = '0: OFF\n 1 : ON'
# cv2.createTrackbar(switch,'image',0,1,nothing)
#
# while(1):
#     cv2.imshow('image',img)
#     k = cv2.waitKey(1) & 0xFF
#     if k ==27:
#         break
#
#     b =cv2.getTrackbarPos('B','image')
#     g = cv2.getTrackbarPos('G', 'image')
#     r = cv2.getTrackbarPos('R', 'image')
#     s = cv2.getTrackbarPos(switch, 'image')
#
#     if  s == 0:
#         img[:] = 0
#     else:
#         img[:] =[b,g,r]
#
#     img[:] = [b,g,r]
#
# cv2.destroyAllWindows()

#-----------------------------------------------------------------------------------------

# Trackbar used on an image
# import cv2
# import numpy as np
#
# def nothing(x):
#     print(x)
#
# # img =np.zeros((300,512,3),np.uint8)
#
# # img = cv2.imread('C:\\Users\cd42146\Downloads\output.JPG',1)
# cv2.namedWindow('image')
#
# cv2.createTrackbar('CP','image',10,400,nothing ) # to creat slide bar
#
# switch = 'color/gray'
# cv2.createTrackbar(switch,'image',0,1,nothing)
#
# while(1):
#     img = cv2.imread('C:\\Users\cd42146\Downloads\output.JPG', 1)
#     img =cv2.imshow('image',img)
#     pos = cv2.getTrackbarPos('CP', 'image')
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(img,str(pos),(50,150),font,10,(0,255,255),10)
#
#     k = cv2.waitKey(1) & 0xFF
#     if k ==27:
#         break
#
#
#     s = cv2.getTrackbarPos(switch, 'image')
#
#
#     if  s == 0:
#         pass
#     else:
#         img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     img = cv2.imread('C:\\Users\cd42146\Downloads\output.JPG', 1)
#
#
#
# cv2.destroyAllWindows()

#----------------------------------------------------------------------------------------------------------------------------------
#
# # HSV ( Hue- color segment (0-360), Saturation(depth of pigment (1-100%)
# #  and Value (brightness of the color (0-100%)
#
# # Masking the color to detect a specific color ball in pic
# # Creating trackbars to set the HUE values .... This is Amazing
# import cv2
# import numpy as np
# #
# def nothing(x):
#      pass
#
# # To capture the video from webcam
# cap = cv2.VideoCapture(0);
#
#
# cv2.namedWindow("Tracking")
# cv2.createTrackbar("LH","Tracking",0,255,nothing)
# cv2.createTrackbar("LS","Tracking",0,255,nothing)
# cv2.createTrackbar("LV","Tracking",0,255,nothing)
# cv2.createTrackbar("UH","Tracking",255,255,nothing)
# cv2.createTrackbar("US","Tracking",255,255,nothing)
# cv2.createTrackbar("UV","Tracking",255,255,nothing)
#
# #
# # img =np.zeros((300,512,3),np.uint8)
# #
# # # img = cv2.imread('C:\\Users\cd42146\Downloads\output.JPG',1)
# # cv2.namedWindow('Tracking')
# # cv2.namedWindow('image')
#
# while True:
#
#     # capture the video frrom webcam
#     _,frame = cap.read()
#
#     # frame = cv2.imread('C:\\Users\cd42146\Downloads\Color Balls.jpg',1)
#     # frame = cv2.imread('C:\\Users\cd42146\Downloads\Tractor with SMV.jpg')
#     hsv =cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#
#     l_h = cv2.getTrackbarPos("LH","Tracking")
#     l_s = cv2.getTrackbarPos("LS", "Tracking")
#     l_v = cv2.getTrackbarPos("LV", "Tracking")
#
#     u_h = cv2.getTrackbarPos("UH", "Tracking")
#     u_s = cv2.getTrackbarPos("US", "Tracking")
#     u_v = cv2.getTrackbarPos("UV", "Tracking")
#
#     l_b = np.array([l_h,l_s, l_v]) # for lower blue layer
#     u_b =np.array([u_h,u_s,u_v])# for upper blue layer
#     mask = cv2.inRange(hsv,l_b,u_b)
#
#     res =cv2.bitwise_and(frame,frame,mask=mask)
#     cv2.imshow("frame",frame)
#     cv2.imshow("frame",mask)
#     cv2.imshow("frame",res)
#
#
#     key = cv2.waitKey(1) & 0xFF
#     if key ==27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()

#---------------------------------------------------------------------------------------------
# # Threshold values
# # If value is more than 125 it will show color or if it is less than that it will show black
# import cv2
# # import numpy as np
#
# img = cv2.imread('C:\\Users\cd42146\Downloads\Color Balls.jpg',1)
# # If value is more than 125 it will show color or if it is less than that it will show
# _, thl =cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# # _, thl =cv2.threshold(img,100,255,cv2.THRESH_TRUNC) # Pixel value remaine same after 100
# # _, thl =cv2.threshold(img,200,255,cv2.THRESH_TOZERO_INV) # Threshold if value less then threshold it notbe zero
#
# # adaptive threshholding where thresh hold didnt make image black below the threshold but it take the neighbourhood brightness
#
# # th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2);
# th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2);
#
# cv2.imshow('Image',img)
# # cv2.imshow('th1',th1)
# # cv2.imshow('th2',th2)
# cv2.imshow('t3',th3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#------------------------------------------------------------------------------------------

# Matplotlib tutorial
#
# import cv2
# from matplotlib import pyplot as plt
#
# img = cv2.imread("C:\\Users\cd42146\Downloads\Color Balls.jpg",-1)
# cv2.imshow('image',img)
# img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)# to convert BGR img to rgb img as matplotlib give the outpu of bgr and open cv give rgb
# plt.imshow(img)
# plt.xticks([]),plt.yticks([]) # to remove the x and y axis values from pic
# plt.show()
#
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#-------------------------------------------------------------------------------

#Morphological to show the image in the binery form using a basic element called kernal
# Dialation , erosion , opening, top hat , morphological gradient .....
# This is amazing too
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# # img = cv2.imread('C:\\Users\cd42146\Downloads\Tractor with SMV.jpg',1)
# img = cv2.imread("C:\\Users\cd42146\Downloads\Color Balls.jpg",cv2.IMREAD_GRAYSCALE)
# _,mask =cv2.threshold(img,220,255,cv2.THRESH_BINARY_INV)
#
# # to remove the small black spots on center of the balls we will use dialation
# # Kernal is the square window which will move over the pic 2x2 size
# kernel = np.ones((2,2),np.uint8) # we can enhance kernal from 2x2 to 5x5 for reducing the dots of the images but the ball area increas
# dilation = cv2.dilate(mask,kernel,iterations=3) # by default iteration value is 1 so if we want
#
# # erosion is like soile erosion concept where side of Object erodes
# eroasion = cv2.erode(mask,kernel,iterations=1)
#
# # opening is another name of erosion followed by dialation
# opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
#
# # dialaiton followed by erosion
#
# closing  = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
#
# mg  = cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,kernel) # morphological gradient - diffrence between dilation and erosion
# th = cv2.morphologyEx(mask,cv2.MORPH_TOPHAT,kernel) # top hat- diffrence between image and its operning
#
#
#
# titles=['image','mask','dilation','erosion','opening','closing','mg','th']
# images =[img,mask,dilation,eroasion,opening,closing,mg,th]
#
# for i in range(8):
#     plt.subplot(4,4,i+1),plt.imshow(images[i],'gray') # 1 row 3 column of image
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()
#------------------------------------------------------------------------------
# Blurring and smoothing the image

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread("C:\\Users\cd42146\Downloads\Color Balls.jpg")
# img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# # homogenious filter have equal weight to each pixel
# # Creating kernal ( a window which slide over say 3x3 over an image)
#
# kernel = np.ones((5,5),np.float32)/25
# dst = cv2.filter2D(img,-1,kernel)
#
# # Low pass filter( LPF) remove the noise from the imnages
# # High  pass filter( HPF) help to detect the edges in the images
# # Blur method - averagign
#
# blur = cv2.blur(img,(8,8)) #sliding 8x8 window
#
# # Gaussian filter is using diffrent - weight - kernal in both x and y directions
# gblur = cv2.GaussianBlur(img,(5,5), 0)
#
# # median filter is that replace each pixel/'s value with the median of its neghbouring pixels .
# # This method is great when dealing with "salt and pepper noise".
#
# median = cv2.medianBlur(img,5) # kernal size should be odd (5 here ) and should not one in median kernal
#
# # bilateral filter - edges are preserved in the much better way while blurring ,
# # effecting in noise removal by keeping edges preserved
# bilateralFilter = cv2.bilateralFilter(img,9,75,75)
#
#
# titles = ['image','2D Convolution','blur','gblur','median','bilateralFilter']
# images = [img,dst,blur,gblur,median,bilateralFilter]
#
# for i in range(6):
#     plt.subplot(3,4,i+1), plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
#
# plt.show()


#------------------------------------------------------------------------------------------------------------------------------------
#
# # Image gradient
# # an image gradient is a directional change in the intensity or color in the image
# # Laplasian method , sobel-x derivative (joint gausian and  diffrentiation operation)
# # sobel-y method
#
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# # img = cv2.imread("C:\\Users\cd42146\Downloads\Color Balls.jpg",cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('C:\\Users\cd42146\Downloads\Tractor with SMV.jpg',cv2.IMREAD_GRAYSCALE)
#
# # laplasing gradient to detect the edges
# lap = cv2.Laplacian(img,cv2.CV_64F,ksize= 3)
# lap = np.uint8(np.absolute(lap)) # to convert laplasian image into unsigned 8 bit image
#
# # Sobel-X , dx- represent the order of the derivative x , dy - represent the order of the derivtive y in below code
# # change in the color intensity in x -direction
# sobelX = cv2.Sobel(img,cv2.CV_64F,1,0)  # when we write 1 means we are using sobel-x method
#
# # Sobel-Y , dx- represent the order of the derivative x , dy - represent the order of the derivtive y in below code
# # change in the color intensity in y -direction
# sobelY = cv2.Sobel(img,cv2.CV_64F,0,1)
#
# sobelX = np.uint8(np.absolute(sobelX)) # to convert sobel image into unsigned 8 bit image
#
# sobelY = np.uint8(np.absolute(sobelY))
#
# sobelCombined = cv2.bitwise_or(sobelX,sobelY)
#
# titles = ['image','Laplasian','sobelX','sobelY','sobelCombined']
# images = [img,lap,sobelX,sobelY,sobelCombined]
#
# for i in range(5):
#     plt.subplot(3,3,i+1), plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
#
# plt.show()

#----------------------------------------------------------------------------------------------------

# # Canny edge detector is an edge detection operator that uses a multi- stage algorithem
# # to detect a wide range of edges in the images . It was developped by John F Canny
# # It is compose of 5 steps ,
# # 1st Noise reduction 2nd Gradient calculation
# # 3rd Non-maximum suppression , 4th Double threshold ,5th edge tracing by hysteresis
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# # img = cv2.imread("C:\\Users\cd42146\Downloads\Color Balls.jpg",0)
# img = cv2.imread('C:\\Users\cd42146\Downloads\Tractor with SMV.jpg',0)
#
# # canny provides less noises and proper edges
# canny = cv2.Canny(img,100,200)  # for threshhold value we have to provide the hyterisis value
#
# titles = ['image','canny']
# images = [img,canny]
#
# for i in range(2):
#     plt.subplot(1,2,i+1), plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
#
# plt.show()
#--------------------------------------------------------------------------------------------
# We can create images of diffrent resolutions by using image pyramade
# pryramid representaiton is a type of multi-scale signal representation in which
# a signal or an image is subject to repeated smoothing and sub sampling
# Two types - Gausian and Laplasian
# laplasian pyramed form out of Gausian pyramed
# # Not exciting
# import cv2
# import numpy as np
#
# img = cv2.imread("C:\\Users\cd42146\Downloads\Color Balls.jpg",0)
# lr1= cv2.pyrDown(img)# to reduce the resolution
# lr2= cv2.pyrUp(lr1) #to increas the resolution
# cv2.imshow("image",img)
# cv2.imshow("image1",lr1)
# cv2.imshow("image2",lr2)
# cv2.waitKey()
# cv2.destroyAllWindows()

#-----------------------------------------------------------------------
#
# # resizing the image
# # didnt worked , leave it , dont use
# import cv2
#
# img1 = cv2.imread('C:\\users\cd42146\Downloads\Apple.jpg', cv2.IMREAD_UNCHANGED)
# img2 = cv2.imread('C:\\users\cd42146\Downloads\Orange.jpg', cv2.IMREAD_UNCHANGED)
# print('Original Dimensions : ', img1.shape)
# print('Original Dimensions : ', img2.shape)
#
# scale_percent = 100  # percent of original size
# width1 = int(img1.shape[1] * scale_percent / 100)
# height1 = int(img1.shape[0] * scale_percent / 100)
# dim1 = (width1, height1)
#
# scale_percent = 60  # percent of original size
# width2 = int(img2.shape[1] * scale_percent / 100)
# height2 = int(img2.shape[0] * scale_percent / 100)
# dim2 = (width2, height2)
#
# # resize image
# resized1 = cv2.resize(img1, dim1, interpolation=cv2.INTER_AREA)
# resized2 = cv2.resize(img2, dim2, interpolation=cv2.INTER_AREA)
#
# print('Resized Dimensions : ', resized1.shape)
# print('Resized Dimensions : ', resized2.shape)
#
# cv2.imshow("Resized image", resized1)
# cv2.imshow("Resized image", resized2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#----------------------------------------------------------------------------
# image blending
# load the two images of same size
# Find the Gaussian Pyramids for images
# From Gaussian Pyramids find their laplacian pyramids
# now join the left half of one pic to right halp of pic in each level of laplasian pyramid
#Finally from this joint image pyramids , reconstruct the orignal image

#
# import cv2
# import numpy as np
#
# apple = cv2.imread('C:\\users\cd42146\Downloads\Apple.jpg', cv2.IMREAD_UNCHANGED)
# orange= cv2.imread('C:\\users\cd42146\Downloads\Orange.jpg', cv2.IMREAD_UNCHANGED)
#
# print(apple.shape)
# print(orange.shape)
#
# # for adding half and half images , images should be equal size
# apple_orange = np.hstack((apple[:,:256],orange[:,256:]))
#
# # Generate gaussian pyramid for apple
# apple_copy = apple.copy()
# gp_apple = [apple_copy]
#
# # to get multiple pyramid of apple image
# for i in range(6):
#     apple_copy= cv2.pyrDown(apple_copy)
#     gp_apple.append(apple_copy)
#
# # Generate gaussian pyramid for orange
# orange_copy = orange.copy()
# gp_orange = [orange_copy]
#
# # to get multiple pyramid of apple image
# for i in range(6):
#     orange_copy= cv2.pyrDown(orange_copy)
#     gp_orange.append(orange_copy)
#
# # generate Laplasian Pyramid for apple
# apple_copy = gp_apple[5]
# lp_apple = [apple_copy]
# for i in range(5,0,-1):
#     gaussian_expanded = cv2.pyrUp(gp_apple[i])
#     laplacian = cv2.subtract(gp_apple[i-1],gaussian_expanded)
#     lp_apple.append(laplacian)
#
# # generate Laplasian Pyramid for orange
# orange_copy = gp_orange[5]
# lp_orange = [orange_copy]
# for i in range(5,0,-1):
#     gaussian_expanded = cv2.pyrUp(gp_orange[i])
#     laplacian = cv2.subtract(gp_orange[i-1],gaussian_expanded)
#     lp_orange.append(laplacian)
#
# # Now add left and right half of the images in each level
#
# apple_orange_pyramid =[]
# n=0
# for apple_lap, orange_lap in zip(lp_apple,lp_orange):
#     n+=1
#     cols ,rows,ch=apple_lap.shape
#     laplacian = np.hstack((apple_lap[:,0,int(cols/2)],orange_lap[:,int(cols/2)]))
#     apple_orange_pyramid.append(laplacian)
# # now reconstruct our image
#
# apple_orange_reconstruct = apple_orange_pyramid[0]
# for i in range(1,6):
#     apple_orange_reconstruct = cv2.pyrUp(apple_orange_reconstruct)
#     apple_orange_reconstruct = cv2.add(apple_orange_pyramid[i],apple_orange_reconstruct)
#
# cv2.imshow("apple",apple)
# cv2.imshow("orange",orange)
# cv2.imshow("apple_orange",apple_orange)
# cv2.imshow("apple_orange_reconstruct",apple_orange_reconstruct)
# cv2.waitKey()
# cv2.destroyAllWindows()

#-----------------------------------------------------------------------------------------------

# Contours - curve made after joining contineous points along the boundy which have same color or boundry
# we use binary image to work with contour
# # used for object detection and object recognition
# # Good to know about the image borders
#
# import numpy as np
# import cv2
# img1 = cv2.imread('C:\\Users\cd42146\Downloads\Tractor with SMV.jpg',0)
# # img2 = cv2.imread('C:\\Users\cd42146\Downloads\Tractor with SMV.jpg',1)
#
# #We are going to apply threshold on canny image
#
# ret, thresh = cv2.threshold(img1,127,250,0)
#
# # now find out the contours
# # Contours is a Python list of all the contours in the image. Each individual
# # Each individual countour is a numpy array of (x,y) coordinates of boundry points of the objects.
# # Hiearchy is an optional output vector which is containing the information about image topologyu
# #
# contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
# print("number of contours = "+ str(len(contours)))
# # when we print the contours we gets x,y cordinates of the contours
# print(contours[10])
# # Now lets join the cordinate to draw contorus
# # now we have to draw those contours on image itself
# cv2.drawContours(img1,contours,-1,(0,255,0),1) # if we will write contouroIndx-1 here , it will draw all the contours
# cv2.imshow('orange',img1)
# # cv2.imshow('orange',img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#-------------------------------------------------------------------------------------
# This is awesome
# Motion detection and trackiing system using Python
#
# import cv2
# import numpy as np
#
# cap = cv2.VideoCapture("C:\\Users\cd42146\Downloads\motion detect.avi")
# #cap = cv2.VideoCapture("C:\\Users\cd42146\Downloads\pose detection sample video.mp4")
# # cap = cv2.VideoCapture("C:\\Users\cd42146\Downloads\VID-20201106-WA0002.mp4")
#
#
# ret,frame1 =cap.read()
# ret,frame2 =cap.read()
#
# while cap.isOpened():
#     # ret,frame = cap.read()
# # absdiff method is to identify diffrence between Frame1 and Frame2 of same video
#     diff = cv2.absdiff(frame1,frame2)
# # once we got diffrence we will convert it in to a gray scale mode
#     gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY) # its easy to find contours in Gray scale mode
# # now just blur our image
#     blur = cv2.GaussianBlur(gray,(3,3),0)
# # find threshold
#     _,thresh = cv2.threshold(blur,20,200,cv2.THRESH_BINARY)
# # Now we will dialate the image to fill all the holes and find out better contours
#     dilated = cv2.dilate(thresh,None,iterations=1)
# # now we will find the contours
#     contours,_, =cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # _ is used as we will not use the second parameter
# # now we will draw the contours
#     # we want to apply contours on orignal frame
#  #   cv2.drawContours(frame1,contours,-1,(0,255,255),2)# ( to draw contours not rectanlges )
#     # how to remove the noices and how to draw rectangles,
#     # so if contour area is greater than some value we would draw a rectangle
#
#
#     for contour in contours:# in this loop we will save the conrdinates of found contours
#         (x,y,w,h) = cv2.boundingRect(contour) # this method is for creating rectangle .now we will find the area of the contour
#
#         if cv2.contourArea(contour) <1200:
#             # it means if the area of the rectangle is less than the 700 , nothing to do
#             continue
#         cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
#         # we want put some text when movement is occurs
#         cv2.putText(frame1,"Status: {}".format('Movement'),(10,20),cv2.FONT_HERSHEY_SIMPLEX,
#                     1,(0,0,255),3)
#
#
#
#     cv2.imshow('feed',frame1)
# # now we will assign values in frame2 in to frame1.
# # we are reading two frames and finding out the diffrence between them
#     frame1 = frame2
#     ret,frame2 = cap.read() # so now we are reading the new variable in frame2
#
#
#     if cv2.waitKey(40)==27:
#         break
#
# cv2.destroyAllWindows()

#--------------------------------------------------------------------------------------------------
# # How we can detect simple geometrical shape and want to write the name top on it
# this is ok but can be very useful for SMV project
# import numpy as np
# import cv2
#
# img = cv2.imread('C:\\Users\cd42146\Downloads\shapes.jpg',0)
# img1 = cv2.resize(img,(650,450),None,250,250,None)
# # settting the threshold value
# _,thresh = cv2.threshold(img1,240,255,cv2.THRESH_BINARY)
#
# # find out the contorus now
# contours ,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#
# for contour in contours:
# # this approxployDP method approximate a polygone curve with a precision
#     approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,closed=True),True) # epsilon is to find approx accurecy. Arclenght calculates contour curve length
#     cv2.drawContours(img1,[approx],0,(0,0,0),2)
# # Now we want to print which shape it is also find the x,y cordinate to print
#     x = approx.ravel()[0] # ravel method
#     y = approx.ravel()[1]-3 # adding offset of 5 will move the text up
# # DPploy method will approx the number of polygone it could be so based on it we can identify which shape it could be
#     if len(approx)==3:
#         cv2.putText(img1,"Triangle",(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0))
#     elif len(approx) == 4:
#         # how to decide its rectangle or not
#         x1,y1,w,h = cv2.boundingRect(approx)
#         aspectRatio =float(w)/h
#         print(aspectRatio)
#         if aspectRatio >0.95 and aspectRatio <= 1.05:
#             cv2.putText(img1, "Square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#         else:
#             cv2.putText(img1, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#     elif len(approx) == 5:
#         cv2.putText(img1, "Pentagone", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#     elif len(approx) == 10:
#         cv2.putText(img1, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#     else:
#         cv2.putText(img1, "Cirle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
#
#
#
#
# cv2.imshow('Shapes',img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#------------------------------------------------------------------------------------------------------------------
# How to detect an Histogram
# Histogram is the plot curve which gives you the idea of the intensity of color in the image
# BGR histogram looks amazing
# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt

# img = np.zeros((200,200), np.uint8) # black pic created
# img = cv.imread('C:\\Users\cd42146\Downloads\Tractor with SMV.jpg',0)
# cv.rectangle(img,(0,100),(200,200),(255),-1)
# cv.rectangle(img,(0,50),(100,100),(127),-1)

# to see histogram in BGR value
# b,g,r = cv.split(img)

#
# cv.imshow("img",img)
# cv.imshow("b",b)
# cv.imshow("g",g)
# cv.imshow("r",r)

# there is another method in open cv to calculate histogram

# hist =cv.calcHist([img],[0],None,[256],[0,256])
# plt.plot(hist)

# because all the value of the pixels are zero so we will get the intensity of pixel zero in histogram
# It help to get idea about contrast , intensity etc of an image
# plt.hist(img.ravel(),256,[0,256])
# plt.hist(b.ravel(),256,[0,256])
# plt.hist(g.ravel(),256,[0,256])
# plt.hist(r.ravel(),256,[0,256])
# plt.show()
#
# cv.waitKey(0)
# cv.destroyAllWindows()

#---------------------------------------------------------------------------
# Template matching open CV python
# Finding an image in the larger image by matching both
# import cv2
# import numpy as np
# img = cv2.imread("C:\\Users\cd42146\Downloads\Tractor with SMV.jpg")
# grey_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# template = cv2.imread("C:\\Users\cd42146\Downloads\SMV Templet.jpg",0)
# w,h = template.shape[::-1] # [::-1] means we want column and rows in reverse order
# #res = cv2.matchTemplate(grey_img,template,cv2.TM_CCOEFF_NORMED) # there are other methods to use like TM_coooEFF_ normed for templet matching
# # Top left corner brightest point in the image search with matching the image templet, print shows it.
# # another method
# res = cv2.matchTemplate(grey_img,template,cv2.TM_CCORR_NORMED)
# print(res)
# # using numpy we can find the brighest point in the print matrix to match and will use 'where' method in numpy
# threshold = 0.8995; # this threshold value provides the left topmost corner which is haveing brightness 0.57 , threshhold value changes as per the TM_CORR methods
# loc = np.where(res>=threshold)
# print(loc)
#
# # if there are multiple matches of the shame in the templet so we can search it in image by using for loop
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img,pt,(pt[0]+w,pt[1]+h),(0,255,255),1)# this will get the bottom right corner of the templet
#
# cv2.imshow("img",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#---------------------------------------------------------------------------------------------
#Hough transform
# To detect any shape , if you can represent that shape in the mathematical form

# import numpy as np
# import cv2
# # from matplotlib import pyplot as plt
#
# # img = np.zeros((200,200), np.uint8) # black pic created
# img = cv2.imread("C:\\Users\cd42146\Downloads\Sudoku.png")
# # conver the image in to grayscale image
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # Detect the edges in the image using canny edge detector
# edges = cv2.Canny(gray,50,150,apertureSize=3)
#
# cv2.imshow('edges',edges)
# lines = cv2.HoughLines(edges,1,np.pi/180,200)
#
# # Hough lines gives the output vector of lines . each line reresetnted in the
# #terms of rho and theta ( polar cordinate systems)
# # we will iterate through each vector of line
# # we will have to convert the polar cordinate ( sin0,cos0 , radians ) in to cartesian cordinate (y=mx+c_
# for line in lines:
#     rho , theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 =a * rho
#     y0 =b * rho
#
#     # x0,y0 are the origin cordinates (0,0) top left corner but we want lines but not the cordinates
#     # so we will have to take other cordinates too
#     # x1 stores the rounded off values of (r*cos(theta)-1000*sin(theta))
#     x1 =int(x0 + 1000 * (-b))
#     # y1 stores the rounded off values of (r*sin(theta)-1000*cos(theta))
#     y1 = int(y0 + 1000 * (a))
#     # x2 stores the rounded off values of (r*cos(theta)+1000*sin(theta))
#     x2 = int(x0 - 1000 * (-b))
#     # y2 stores the rounded off values of (r*sin(theta)-1000*cos(theta))
#     y2 = int(y0 - 1000 * (a))
#
#     # Line method takes cartesian cordinate. We need atleast two cordinates to make a line
#     # x1,y1 and x2,y2 are the cordinates to draw the lines
#     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2) # 2 is the thickness of the line
#
# cv2.imshow('image',img)
# k= cv2.waitKey(0)
# cv2.destroyAllWindows()
# in this hough line you will see the lines are going infinite or one ecd to anothr end of picture
#------------------------------------------------------------------------------------------
# # we will hough line P to eleminate it.
# import numpy as np
# import cv2
# # from matplotlib import pyplot as plt
#
# # img = np.zeros((200,200), np.uint8) # black pic created
# img = cv2.imread("C:\\Users\cd42146\Downloads\Sudoku.png")
# # conver the image in to grayscale image
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # Detect the edges in the image using canny edge detector
# edges = cv2.Canny(gray,50,150,apertureSize=3)
#
# cv2.imshow('edges',edges)
# lines = cv2.HoughLinesP(edges,1,np.pi/180,100,lines=None,minLineLength=100,maxLineGap=10)
# for line in lines:
#     x1,y1,x2,y2 = line[0]
#     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),0)
#
#
# cv2.imshow('image',img)
# k= cv2.waitKey()
# cv2.destroyAllWindows()

#----------------------------------------------------------------------------------------------

# Lane detected system in a video , it works perect ,awesome


# Now we will apply line detection on the video
# Video containes many imaages as frames
# It works well
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
#
# # image = cv2.imread("C:\\Users\cd42146\Downloads\Road.jpeg")
# # image = cv2.imread("C:\\Users\cd42146\Downloads\Road1.jpg")
# # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#
#
#
# def region_of_interest(img,vertices):
#     mask = np.zeros_like(img)
#     # channel_count = img.shape[2] #  we would not need any color channel because its gray scale image
#     # now are going to create match color with same  same color count
#     match_mask_color = 255 # for maskin the image we will need only on channel so make it 255
#     # Now we are going to fill the polygone using fillpoly method because we want to fill everything apart from our region of interst
#     cv2.fillPoly(mask,vertices,match_mask_color)
#     #next image we are going to return only those pixels where the mask pixel matches
#     mask_image = cv2.bitwise_and(img,mask)
#     return mask_image
# # Define the region of the interest
# # all the lane are parallel and will seems merging as some point of time
# # Our vehicle will only drive on one side and rest other object are noise
# #
# # function to draw the hough transoform lines
#
# def draw_the_lines(img,lines):
#      img =np.copy(img) # copying the image
#      # now we are creating a blank image which will matched with orignal image size
#      blank_image = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8) # img.shape[0] is the height,img.shape[1] is the weight ,3- is number of channels
#     # now we are going to loop around line vectors and draw the line
#      for line in lines:
#          for x1,y1,x2,y2 in line:
#             cv2.line(blank_image,(x1,y1),(x2,y2),(0,255,0),thickness=3)
# # we want to draw lines on the blank image and then merge it with orignal image
# # merging the image It will give us the line drawn on orignal image
#
#      img = cv2.addWeighted(img,0.8,blank_image,1,0.0)
#      # once we will have line on the image we will return it
#
#      return img
#
# def process(image):
#     # print(image.shape)
#     height = image.shape[0]
#     width = image.shape[1]
#
#     # region of the interst will be made of 3 points  (first (left corner where width is zero but height)
#     # second point at middle where length and height are half and
#     # 3rd the lowe right corner
#     region_of_intrest_vertices = [
#         (0,height),
#         (width/2, height/2),
#         (width,height)]
#
#     # Now we will create a function which will mask all other function apart from the region of interest
#
#     gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
#     canny_image = cv2.Canny(gray_image,100,120)
#
#     cropped_image = region_of_interest(canny_image,
#                     np.array([region_of_intrest_vertices],np.int32),)
#     # with above code we have identify the region of the interest and now find the edges
#     # now we will be using hough line transform
#     # for that we will have to convert the image in to the gray scale image
#     # we were getting the region of interest border as edges in the image
#
#     # now it would be easy to plot hough line transform on cropped canny edged image
#
#     lines = cv2.HoughLinesP(cropped_image,rho=2,
#                             theta=np.pi/60,
#                             threshold=50,
#                             lines=np.array([]),
#                             minLineLength=40,
#                             maxLineGap=100)
#     # after applying hough transfor it is going to return hough line vectors detected within our image
#     # now we will make a funciton to draw the lines
#
#
#
#     image_with_lines  = draw_the_lines(image,lines)
#     return image_with_lines # it means we are going to draw the lines on every image of this function
#
# cap = cv2.VideoCapture("C:\\Users\cd42146\Downloads\Road video.mp4")
# # check the frames are availeble as while loop
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     frame = process(frame)
#     # here we will process each frame of the video with using process function
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(100) & 0xFF ==ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

#-------------------------------------------------------------------------

# Hough Transform

# circle transform
# import numpy as np
# import cv2 as cv
#
# # img = cv.imread("C:\\Users\cd42146\Downloads\Color Balls.jpg")
# img = cv.imread("C:\\Users\cd42146\Downloads\shapes.jpg")
# output = img.copy()
# # Circle formula (x-xcenter)**2 + (y-ycenter)**2 = r**2
# # so if we know (x,y)cordinates and r- radius , we can draw a circle
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# gray = cv.medianBlur(gray,5)
#
#
#
# #dp - inverse ratio of the accumulatior resolution to the image resolutiqon
# #minDist - min. distance between the centers of the detect circles
# #param1 -First method -specofic parameters. In care of HOUGH_GRADIENT, it is
# #the hight threshold of the two passed to the Canny edge detector( the
# #lower one is twise smaller).
# # param2 - Second method -specific parameter. In case of HOUGH_GRADIENT , its is the accumulator
# #threshold for the circle center at the detection stage.
# #minRadius - Minimum Circle Radius
# #maxRadius - Maximum circle radius if <=0 usese the maximum image dimension
# # if <0 returns center without finding radius
# circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,20,circles=None
#                           ,param1=60,param2=30,
#                           minRadius=0,maxRadius=0)
# # we will have to convert the parameters what we got from circle variable
# # in to i.e. x & y cordinate and radius in to integer
#
# detected_circle = np.uint16(np.around(circles)) # this vector will give us x,y, radius so extract these values
#
# for (x,y,r) in detected_circle[0,:]:
#     cv.circle(output,(x,y),r,(0,255,0),3)
#     cv.circle(output, (x, y), 2, (0, 255, 255), 3) # radius 2 is small to draw the circle
#
#
#
# cv.imshow('output',output)
# cv.waitKey(0)
# cv.destroyAllWindows()

#----------------------------------------------------------------------------------

# HAR Cascade based face detection , Object detection harcascade classifire
# Cascade function has been trained on lots of posetive (image containes the object we want in it)  and negetive images
# Once classifire trained on +ve and -ve images than it find region of interest
# import cv2
#
# # import the haar classifire
# face_cascade = cv2.CascadeClassifier('C:\\Users\cd42146\Downloads\haarcascade_frontalface_default.txt')
# # img = cv2.imread('C:\\Users\cd42146\Downloads\Color Balls.jpg')
# img = cv2.imread('C:\\Users\cd42146\Downloads\Human Faces.jpg')
# # convert the image in to gray image
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# # Below - scaleFactor - how much the image size is reduced at each image scale
# # minNeighbours - parameters specifying how many neighbors each candidate rectangle should have to retain it.
#
# faces = face_cascade.detectMultiScale(gray,1.1,4)
#
# # now we want to iterate over all the faces and give a rectangle
#
# for (x,y,w,h) in faces:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
#
#
# cv2.imshow('img', img)
# cv2.waitKey()

#_____________________________________________________________________________________
# # If we want to apply it on any Video for that:
#
# import cv2
#
# # import the haar classifire
# face_cascade = cv2.CascadeClassifier('C:\\Users\cd42146\Downloads\haarcascade_frontalface_default.txt')
# # img = cv2.imread('C:\\Users\cd42146\Downloads\Color Balls.jpg')
# # img = cv2.imread('C:\\Users\cd42146\Downloads\Human Faces.jpg')
# # convert the image in to gray image
#
# cap = cv2.VideoCapture(0) # zero for webcam
#
# while cap.isOpened():
#     _,img =cap.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # Below - scaleFactor - how much the image size is reduced at each image scale
#     # minNeighbours - parameters specifying how many neighbors each candidate rectangle should have to retain it.
#
#     faces = face_cascade.detectMultiScale(gray, 1.1, 4)
#
#     # now we want to iterate over all the faces and give a rectangle
#
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
#
#     cv2.imshow('img', img)
#     cv2.waitKey()
#     if cv2.waitKey(1) & OxFF == ord('q'):
#         break
#
# cap.release()

#______________________________________________________________________________

# # how to detect the eyes using Haar cascade
# import cv2
#
# # import the haar classifire
# face_cascade = cv2.CascadeClassifier('C:\\Users\cd42146\Downloads\haarcascade_frontalface_default.txt')
# eye_cascade = cv2.CascadeClassifier('C:\\Users\cd42146\Downloads\haarcascade_eye_tree_eyeglasses.txt')
# # img = cv2.imread('C:\\Users\cd42146\Downloads\Color Balls.jpg')
# # img = cv2.imread('C:\\Users\cd42146\Downloads\Human Faces.jpg')
# # convert the image in to gray image
#
# cap = cv2.VideoCapture(0) # zero for webcam
#
# while cap.isOpened():
#     _,img =cap.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # Below - scaleFactor - how much the image size is reduced at each image scale
#     # minNeighbours - parameters specifying how many neighbors each candidate rectangle should have to retain it.
#
#     faces = face_cascade.detectMultiScale(gray, 1.1, 4)
#
#     # now we want to iterate over all the faces and give a rectangle
#
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
#         # eyes will be in the area of the interest of face so not we are going to create ROI
#         roi_gray = gray[y:y+h, x:x+w] # we just want face out of the gray scale image
#         roi_color = img[y:y + h, x:x + w]# we also want color of image too
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#         # now we will agin iterate over eyes
#         for (ex,ey,ew,eh) in eyes:
#             cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),5)
#
#     cv2.imshow('img', img)
#     cv2.waitKey()
#     if cv2.waitKey(1) & OxFF == ord('q'):
#         break
#
# cap.release()

#------------------------------------------------------------------------------------------------

# This is very useful to detect the corners of the Triangle
# Harris Corner Detector
#
# Containes 3 main steps
# 1. Determine which window produce very large variation in the intencity when moved both x& y direction
# with each such window found , a score R is computed
# 3. After applying a threshold to this score , important corners are selected and marked


import  numpy as np
import cv2 as cv

# img = cv.imread("C:\\Users\cd42146\Downloads\Color Balls.jpg")
# img = cv.imread("C:\\Users\cd42146\Downloads\Sudoku.png")
img = cv.imread('C:\\Users\cd42146\Downloads\Tractor with SMV.jpg')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

gray = np.float32(gray)

# corner Harris method takes the gray scale image and that should be in
# floating point image else corner harris algorithem does not work
# img - input image , it should be gray scale image and float32 type
# Blocksize - it is the size of the neighbourhood considered for corner detection
# kSize - Aperture parameters of the Sobel derivative used.
# K - Harris detector free parameter in the equation

dst = cv.cornerHarris(gray,2,3,0.05) # 2- is the block size means each neighbourhood 2x2 block size is considered

dst = cv.dilate(dst ,None)

# detecting the image with threshold value and marking it with the colour
img[dst> 0.01*dst.max()]=[0,255,0]

cv.imshow('dst',img)

if cv.waitKey(0) & 0xff ==27:
    cv.destroyAllWindows()

#----------------------------------------------------------------------------------------
#Shi Tomashi corner detector - an imporved version of the corner detector.

# apart from all  Harris corner the value of r calculated in
# Good features to track method
# This also reduce the unwanted corners
#
# import  numpy as np
# import cv2 as cv
#
# # img = cv.imread("C:\\Users\cd42146\Downloads\Color Balls.jpg")
# # img = cv.imread("C:\\Users\cd42146\Downloads\Sudoku.png")
# img = cv.imread('C:\\Users\cd42146\Downloads\Tractor with SMV.jpg')
#
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#
# # paper published by she-tomashi was Good features to track
# # 25- max no. of corners (most importatn) , 0.01 is quality of the corners
# # 10 is the max. equladian distance of the corner
# corners = cv.goodFeaturesToTrack(gray,550,0.05,4)
#
# # once corner detected we convert those corners in integer values
# # int0 is mere alias of int64
# corners = np.int0(corners)
#
# # in order to detect all the corners we will have to iterate over all the corncers
# for i in corners:
#     x,y = i.ravel() # ravel flaten the multidimensonal arra (3x2) to (6x1)
#     # to draw the circles over the corners
#     cv.circle(img,(x,y),3,255,-1)
#
# cv.imshow('dst',img)
#
# if cv.waitKey(0) & 0xff ==27:
#     cv.destroyAllWindows()
#-------------------------------------------------------------------------------------

# Background Substraction Method
# Background mask calculate foreground mask substraction
# Can be used in vistor room calculting no. of visitor coming or leaving the room or
# for traffic vehicle car from moving traffic

# import numpy as np
# import cv2 as cv
#
# cap = cv.VideoCapture("C:\\Users\cd42146\Downloads\motion detect.avi")
#
# # Define a variable, fore ground and backgraound
# # Its a gausian mixture based segmentation algorithem , MOG2 does detect the shadow
# # Detect shadow is an optional parameter to provide
# # This is first method
# # fgbg= cv.createBackgroundSubtractorMOG2(detectShadows=True)
#
# # this second method to extra the background
#
# fgbg= cv.createBackgroundSubtractorKNN()
#
# while True:
#     ret,frame = cap.read()
#     if frame is None :
#         break
#     #creating a foreground mask
#     fgmask =fgbg.apply(frame)
#ap = cv2.VideoCapture("C:\\Users\cd42146\Downloads\Road video.mp4")
#     cv.imshow('Frame', frame)
#     cv.imshow('FG Mask Frame', fgmask)
#
#     keyboard = cv.waitKey(30)
#     if keyboard =='q' or keyboard ==27:
#         break
# cap.release()
#
# cv.destroyAllWindows()

#-----------------------------------------------------------------------------------------------------------------------------------------------

#Object Tracking-locating a omoving object over time
#
# # Mean shift algorithem  - moving the area (window) toward maximum density of pixel points
# import numpy as np
# import cv2
#
# cap = cv2.VideoCapture("C:\\Users\cd42146\Downloads\highway.mp4")
# # Mean shift algorithem steps
#
# # 1st - take first frame of the video
# ret,frame = cap.read() # this will give us the first frame of the video
#
# # 2nd -setup initial location of the window, here in code it is already calculated
# x,y ,width,height = 300,200,100,50
# track_window = (x,y,width,height)
#
# #3rd -set up the ROI (region of interest) for the tracking
# # also need to calculate the histograhm back projection
# roi = frame[y:y+height,x:x+width]
#
# # cv2.imshow('roi',roi)
# # 4th - set up the termination criteria , either 10 iteration or move atleast by 1 point
# # for histogram only hsv channel works
# hsv_roi =cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
# # to avoide low light values we used the in range function
# mask =cv2.inRange(hsv_roi,np.array((0.,60.,32.)),np.array((180.,255.,255.)))
# roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
# cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
# term_crit ={cv2.TERM_CRITERIA_COUNT,10,1}
# # above all will give us a histogram pad projected image
# # once we got the histograhm moving object we will move it over the video
# # while(cap.isOpened()):
# #     ret, frame = cap.read()
# #
# #     # here we will process each frame of the video with using process function
# #     cv2.imshow('frame',frame)
# #     if cv2.waitKey(100) & 0xFF ==ord('q'):
# #         break
# # cap.release()
# # cv2.destroyAllWindows()
#
# while(1):
#     ret,frame = cap.read()
#     if ret == True:
#
#         hsv =cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#
#         # now we will use a function calculte back project
#         dst =cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
#
#         # apply mainshift to get the new location.
#         # now we will provide new track picture , term_crit - we will have to define out side of the loop
#         ret,track_window = cv2.meanShift(dst,track_window,term_crit)
#
#         #draw it on image
#         x,y,w,h = track_window
#         final_image = cv2.rectangle(frame,(x,y),(x+w),(y+h),255,3)
#
#         # cv2.imshow('frame',frame)
#         cv2.imshow('final_image',final_image)
#
#         # if you want to see back projected image
#         cv2.imshow('dst', dst)
#         k=cv2.waitKey(30) & 0xFF
#         if k ==27:
#             break
#     cap.release()
#     cv2.destroyAllWindows()

#The disadvantage of the meanshift is that the size of the window remain same

# even if object moves closer and closer
#  We have to give the initial region of the interest and
# if initial position of the window now known its not easy to apply mean shift method
#-------------------------------------------------------------------------------------------
# CAM shift - Contineous adaptive mean shift
# so if object moves CAM shift change the size of window and adapt the
# best fitting orientation of the window to it
#
# # Note below code didnt works
# import numpy as np
# import cv2
#
# cap = cv2.VideoCapture("C:\\Users\cd42146\Downloads\highway.mp4")
# # Mean shift algorithem steps
#
# # 1st - take first frame of the video
# ret,frame = cap.read() # this will give us the first frame of the video
#
# # 2nd -setup initial location of the window, here in code it is already calculated
# x,y ,width,height = 300,200,100,50
# track_window = (x,y,width,height)
#
# #3rd -set up the ROI (region of interest) for the tracking
# # also need to calculate the histograhm back projection
# roi = frame[y:y+height,x:x+width]
#
# # cv2.imshow('roi',roi)
# # 4th - set up the termination criteria , either 10 iteration or move atleast by 1 point
# # for histogram only hsv channel works
# hsv_roi =cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
# # to avoide low light values we used the in range function
# mask =cv2.inRange(hsv_roi,np.array((0.,60.,32.)),np.array((180.,255.,255.)))
# roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
# cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
# term_crit ={cv2.TERM_CRITERIA_COUNT| cv2.TERM_CRITERIA_EPS,10,1}
# cv2.imshow('roi',roi)
# # above all will give us a histogram pad projected image
# # once we got the histograhm moving object we will move it over the video
# # while(cap.isOpened()):
# #     ret, frame = cap.read()
# #
# #     # here we will process each frame of the video with using process function
# #     cv2.imshow('frame',frame)
# #     if cv2.waitKey(100) & 0xFF ==ord('q'):
# #         break
# # cap.release()
# # cv2.destroyAllWindows()
#
# while(1):
#     ret,frame = cap.read()
#     if ret == True:
#
#         hsv =cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#
#         # now we will use a function calculte back project
#         dst =cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
#
#         # apply mainshift to get the new location.
#         # now we will provide new track picture , term_crit - we will have to define out side of the loop
#         ret,track_window = cv2.meanShift(dst,track_window,term_crit)
#         print(ret)
#         #draw it on image
#         #PTS is for rotating rectangle over image
#
#         pts = cv2.boxPoints(ret)
#         print(pts)
#         pts = np.int0(pts)
#         final_image =cv2.polylines(frame,[pts],True,[0,255,0],2)
#         x,y,w,h = track_window
#         final_image = cv2.rectangle(frame,(x,y),(x+w),(y+h),255,3)
#
#         # cv2.imshow('frame',frame)
#         cv2.imshow('final_image',final_image)
#
#         # if you want to see back projected image
#         cv2.imshow('dst', dst)
#         k=cv2.waitKey(30) & 0xFF
#         if k ==27:
#             break
#         cap.release()
#         cv2.destroyAllWindows()