import cv2
import numpy as np
# import smtplib
import playsound
import threading
#
Alarm_Status = False
Email_Status = False
Fire_Reported = 0


#
def play_alarm_sound_function():
    while True:
        playsound.playsound('C:\\Users\cd42146\Downloads\Alarm Sound.mp3', True)


# def send_mail_function():
#     recipientEmail = "Enter_Recipient_Email"
#     recipientEmail = recipientEmail.lower()
#
#     try:
#         server = smtplib.SMTP('smtp.gmail.com', 587)
#         server.ehlo()
#         server.starttls()
#         server.login("Enter_Your_Email (System Email)", 'Enter_Your_Email_Password (System Email')
#         server.sendmail('Enter_Your_Email (System Email)', recipientEmail,
#                         "Warning A Fire Accident has been reported on ABC Company")
#         print("sent to {}".format(recipientEmail))
#         server.close()
#     except Exception as e:
#         print(e)
# If you want to use webcam use Index like 0,1.

video = cv2.VideoCapture("C:\\Users\cd42146\Downloads\Burning wood in firepit.mp4")



while True:
    (grabbed, frame) = video.read()
    if not grabbed:
        break

    frame = cv2.resize(frame, (960, 540))

    blur = cv2.GaussianBlur(frame, (21, 21), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower = [18, 50, 50]
    upper = [35, 255, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    mask = cv2.inRange(hsv, lower, upper)

    output = cv2.bitwise_and(frame, hsv, mask=mask)

    no_red = cv2.countNonZero(mask)

    if int(no_red) > 15000:
        Fire_Reported = Fire_Reported + 1

    cv2.imshow("output", output)

    if Fire_Reported >= 1:

        if Alarm_Status == False:
            threading.Thread(target=play_alarm_sound_function).start()
            Alarm_Status = True

        # if Email_Status == False:
        #     threading.Thread(target=send_mail_function).start()
        #     Email_Status = True

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video.release()


###############################################################################
 # To make on the camera of Respi
# import cv2
# #
# # cap =cv2.VideoCapture(0)
# #
# # while True:
# #     ret, frame = cap.read()
# #
# #     cv2.imshow("Cam",frame)
# #
# #     if cv2.waitKey(1)==13:
# #         break
# # cap.release()
# # cv2.destroyAllWindows()


#################LED blinking with Respi###############
#
# import RPI.GPIP as GPIO
# import time
# GPIO.setmode(GPIO.BOARD)
# GPIO.setup(3,GPIO.OUT)
#
#
# while True:
#     GPIO.output(3,True)
#     time.sleep(1)
#     GPIO.output(3,False)
#     time.sleep(1)


#### ResPi interface with IR sensor

## to read the sensor data

import RPI.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setup(3,GPIO.IN)
GPIO.setup(5,GPIO.OUT) ### for pump or any other dvise to start
while True:
    val = GPIO.input(3)
    Print(val)
    if val==1:
        GPIO.output(5,GPIO.LOW)
    else:
        GPIO.output(5,GPIO.HIGH)


