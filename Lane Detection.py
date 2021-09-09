# Road Lane detection
# Region of interest find it out


# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
#
# # image = cv2.imread("C:\\Users\cd42146\Downloads\Road.jpeg")
# image = cv2.imread("C:\\Users\cd42146\Downloads\Road1.jpg")
# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#
# # Define the region of the interest
# # all the lane are parallel and will seems merging as some point of time
# # Our vehicle will only drive on one side and rest other object are noise
#
# print(image.shape)
# height = image.shape[0]
# width = image.shape[1]
#
# # region of the interst will be made of 3 points  (first (left corner where width is zero but height)
# # second point at middle where length and height are half and
# # 3rd the lowe right corner
# region_of_intrest_vertices = [ (0,height),(width/2, height/2),(width,height)]
#
# # Now we will create a function which will mask all other function apart from the region of interest
#
# def region_of_interest(img,vertices):
#     mask = np.zeros_like(img)
#     channel_count = img.shape[2] # we will be retrive the number of channels from image (w,h,channels) so index [2]
#     # now are going to create match color with same  same color count
#     match_mask_color = (255,)*channel_count
#     # Now we are going to fill the polygone using fillpoly method because we want to fill everything apart from our region of interst
#     cv2.fillPoly(mask,vertices,match_mask_color)
#     #next image we are going to return only those pixels where the mask pixel matches
#     mask_image = cv2.bitwise_and(img,mask)
#     return mask_image
#
#
# cropped_image = region_of_interest(image,
#                 np.array([region_of_intrest_vertices],np.int32),)
# # with above code we have identify the region of the interest and now find the edges
# # now we will be using hough line transform
#
# plt.imshow(cropped_image)
# plt.show()

#--------------------------------------------------------------------------------------

# This is importatnt but not working well !
# the picture settings changes the performance of the algorithem
#
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
#
# # image = cv2.imread("C:\\Users\cd42146\Downloads\Road.jpeg")
# image = cv2.imread("C:\\Users\cd42146\Downloads\Road1.jpg")
# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
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
# Define the region of the interest
# all the lane are parallel and will seems merging as some point of time
# Our vehicle will only drive on one side and rest other object are noise

# function to draw the hough transoform lines
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
#
# print(image.shape)
# height = image.shape[0]
# width = image.shape[1]
#
# # region of the interst will be made of 3 points  (first (left corner where width is zero but height)
# # second point at middle where length and height are half and
# # 3rd the lowe right corner
# region_of_intrest_vertices = [
#     (0,height),
#     (width/2, height/2),
#     (width,height)]
#
# # Now we will create a function which will mask all other function apart from the region of interest
#
# gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
# canny_image = cv2.Canny(gray_image,100,200)
#
# cropped_image = region_of_interest(canny_image,
#                 np.array([region_of_intrest_vertices],np.int32),)
# # with above code we have identify the region of the interest and now find the edges
# # now we will be using hough line transform
# # for that we will have to convert the image in to the gray scale image
# # we were getting the region of interest border as edges in the image
#
# # now it would be easy to plot hough line transform on cropped canny edged image
#
# lines = cv2.HoughLinesP(cropped_image,rho=6,
#                         theta=np.pi/60,threshold=160,
#                         lines=np.array([]),
#                         minLineLength=40,
#                         maxLineGap=25)
# # after applying hough transfor it is going to return hough line vectors detected within our image
# # now we will make a funciton to draw the lines
#
#
#
# image_with_lines  = draw_the_lines(image,lines)
# plt.imshow(image_with_lines)
# plt.show()

#--------------------------------------------------------------------------------------------------

# Now we will apply line detection on the video
# Video containes many imaages as frames
# It works well
import matplotlib.pyplot as plt
import cv2
import numpy as np

# image = cv2.imread("C:\\Users\cd42146\Downloads\Road.jpeg")
# image = cv2.imread("C:\\Users\cd42146\Downloads\Road1.jpg")
# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)



def region_of_interest(img,vertices):
    mask = np.zeros_like(img)
    # channel_count = img.shape[2] #  we would not need any color channel because its gray scale image
    # now are going to create match color with same  same color count
    match_mask_color = 255 # for maskin the image we will need only on channel so make it 255
    # Now we are going to fill the polygone using fillpoly method because we want to fill everything apart from our region of interst
    cv2.fillPoly(mask,vertices,match_mask_color)
    #next image we are going to return only those pixels where the mask pixel matches
    mask_image = cv2.bitwise_and(img,mask)
    return mask_image
# Define the region of the interest
# all the lane are parallel and will seems merging as some point of time
# Our vehicle will only drive on one side and rest other object are noise
#
# function to draw the hough transoform lines

def draw_the_lines(img,lines):
     img =np.copy(img) # copying the image
     # now we are creating a blank image which will matched with orignal image size
     blank_image = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8) # img.shape[0] is the height,img.shape[1] is the weight ,3- is number of channels
    # now we are going to loop around line vectors and draw the line
     for line in lines:
         for x1,y1,x2,y2 in line:
            cv2.line(blank_image,(x1,y1),(x2,y2),(0,255,0),thickness=3)
# we want to draw lines on the blank image and then merge it with orignal image
# merging the image It will give us the line drawn on orignal image

     img = cv2.addWeighted(img,0.8,blank_image,1,0.0)
     # once we will have line on the image we will return it

     return img

def process(image):
    # print(image.shape)
    height = image.shape[0]
    width = image.shape[1]

    # region of the interst will be made of 3 points  (first (left corner where width is zero but height)
    # second point at middle where length and height are half and
    # 3rd the lowe right corner
    region_of_intrest_vertices = [
        (0,height),
        (width/2, height/2),
        (width,height)]

    # Now we will create a function which will mask all other function apart from the region of interest

    gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image,100,120)

    cropped_image = region_of_interest(canny_image,
                    np.array([region_of_intrest_vertices],np.int32),)
    # with above code we have identify the region of the interest and now find the edges
    # now we will be using hough line transform
    # for that we will have to convert the image in to the gray scale image
    # we were getting the region of interest border as edges in the image

    # now it would be easy to plot hough line transform on cropped canny edged image

    lines = cv2.HoughLinesP(cropped_image,rho=2,
                            theta=np.pi/60,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)
    # after applying hough transfor it is going to return hough line vectors detected within our image
    # now we will make a funciton to draw the lines



    image_with_lines  = draw_the_lines(image,lines)
    return image_with_lines # it means we are going to draw the lines on every image of this function

cap = cv2.VideoCapture("C:\\Users\cd42146\Downloads\Road video.mp4")
# check the frames are availeble as while loop
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = process(frame)
    # here we will process each frame of the video with using process function
    cv2.imshow('frame',frame)
    if cv2.waitKey(100) & 0xFF ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
