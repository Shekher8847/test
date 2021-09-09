# import cv2
#
# # define a video capture object
# vid = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#
# while (True):
#
#     # Capture the video frame
#     # by frame
#     ret, frame = vid.read()
#
#     # Display the resulting frame
#     cv2.imshow('frame', frame)
#
#     # the 'q' button is set as the
#     # quitting button you may use any
#     # desired button of your choice
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # After the loop release the cap object
# vid.release()
# # Destroy all the windows
# cv2.destroyAllWindows()




################################################################################################

# def show_webcam(mirror=False, width=600, height=600):
# #     cam = cv2.VideoCapture(0)
# #     while True:
# #         ret_val, img = cam.read()
# #         if mirror:
# #             img = cv2.flip(img, 1)
# #         cv2.imshow('my webcam', img)
# #         cv2.namedWindow('my webcam',cv2.WINDOW_NORMAL)
# #         cv2.resizeWindow('my webcam', width, height)
# #         if cv2.waitKey(1) == 27:
# #             break  # esc to quit
# #     cv2.destroyAllWindows()
#############################################################################################################

camera_port = 0
import cv2, time
1#1. Create an object , zero for external camera
video = cv2.VideoCapture(camera_port,cv2.CAP_DSHOW)

a =0
while True:
    a = a +1

    #3. Create a frame object

    check,frame  = video.read()

    print(check)
    print(frame) # Representing image


    #6. Converting grey scale

      #gray =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #show the frames !

    cv2.imshow("Capturing",frame)

    #5. fro press any key to out (milliseconds)

    cv2.waitKey(0)

    # for  playing

    key =cv2.waitKey(1)


    if key == ord('q'):
        break

print(a)

#2.Shutdown the camera
video.release()

cv2.destroyAllWindows()




###################################################################
import cv2, time

# camera_port = 0
# #camera = cv2.VideoCapture(camera_port)
# camera = cv2.VideoCapture(camera_port,cv2.CAP_DSHOW)
# # Check if the webcam is opened correctly
# if not camera.isOpened():
#     raise IOError("Cannot open webcam")
#
# return_value, image = camera.read()
# print("We take a picture of you, check the folder")
# cv2.imwrite("image.png", image)
#
# camera.release() # Error is here
# cv2.destroyAllWindows()

#
# camera_port = 0
# camera = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW) # Added cv2.CAP_DSHOW
# return_value, image = camera.read()
# cv2.imwrite("image.png", image)
# print(image)
#
# camera.release()
# cv2.destroyAllWindows() # Handles the releasing of the camera accordingly