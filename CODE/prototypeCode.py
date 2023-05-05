import numpy as np
from picamera import PiCamera
import cv2
from simple_pid import PID
from gpiozero import Servo
from time import sleep
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Device
import math
import subprocess

Xval = 0
Yval = 0
servoOutputX = 0
servoOutputY = 0
Xcenter = 0.075
Ycenter = -0.04
SPX = 80
SPY = 80
subprocess.Popen(["pigpiod"])

kernel = np.ones((5,5),np.uint8)

# Take input from webcam
cap = cv2.VideoCapture(-1)

# Reduce the size of video to 320x240 so rpi can process faster
cap.set(3,SPX*2)
cap.set(4,SPY*2)

def nothing(x):
    pass
# Creating a windows for later use
cv2.namedWindow('XPID')
cv2.namedWindow('YPID')
cv2.namedWindow('tracking')


# Creating track bar for min and max for hue, saturation and value
# You can adjust the defaults as you like
cv2.createTrackbar('P', 'XPID',150,200,nothing)
cv2.createTrackbar('I', 'XPID',190,200,nothing)
cv2.createTrackbar('D', 'XPID',170,200,nothing)
cv2.createTrackbar('P', 'YPID',150,200,nothing)
cv2.createTrackbar('I', 'YPID',190,200,nothing)
cv2.createTrackbar('D', 'YPID',170,200,nothing)

Device.pin_factory = PiGPIOFactory()


servoX = Servo(25)
servoY = Servo(21)

pidX = PID(0.0, 0.0, 0.0, setpoint=SPX)
pidX.output_limits = (-0.06, 0.06)
pidY = PID(0.0, 0.0, 0.0, setpoint=SPY)
pidY.output_limits = (-0.06, 0.06)

while(1):

    buzz = 0
    _, frame = cap.read()

    #converting to HSV
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    hue,sat,val = cv2.split(hsv)

    hmn = 30
    hmx = 92


    smn = 8
    smx = 136


    vmn = 205
    vmx = 255

    # get info from track bar and appy to result
    Xp = cv2.getTrackbarPos('P','XPID')
    Xi = cv2.getTrackbarPos('I','XPID')
    Xd = cv2.getTrackbarPos('D','XPID')
    Yp = cv2.getTrackbarPos('P','YPID')
    Yi = cv2.getTrackbarPos('I','YPID')
    Yd = cv2.getTrackbarPos('D','YPID')


    # Apply thresholding
    hthresh = cv2.inRange(np.array(hue),np.array(hmn),np.array(hmx))
    sthresh = cv2.inRange(np.array(sat),np.array(smn),np.array(smx))
    vthresh = cv2.inRange(np.array(val),np.array(vmn),np.array(vmx))

    # AND h s and v
    tracking = cv2.bitwise_and(hthresh,cv2.bitwise_and(sthresh,vthresh))

    # Some morpholigical filtering
    dilation = cv2.dilate(tracking,kernel,iterations = 1)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    #closing = cv2.GaussianBlur(closing,(5,5),0)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(closing,cv2.HOUGH_GRADIENT,2,120,param1=120,param2=50,minRadius=10,maxRadius=0)
    # circles = np.uint16(np.around(circles))

    #Draw Circles
    if circles is not None:
            for ii in circles[0,:]:
                # If the ball is far, draw it in green
                print (ii[0], ii[1])
                Xval = ii[0]
                Yval = ii[1]

cv2.circle(frame,(int(round(ii[0])),int(round(ii[1]))),2,(0,255,0),10)

        #you can use the 'buzz' variable as a trigger to switch some GPIO
lines on Rpi :)
    # print buzz
    # if buzz:
        # put your GPIO line here


    XmyP = float(Xp)/1000.0-0.2
    XmyI = float(Xi)/1000.0-0.2
    XmyD = float(Xd)/1000.0-0.2
    print(XmyP, XmyI, XmyD)
    YmyP = float(Yp)/1000.0-0.2
    YmyI = float(Yi)/1000.0-0.2
    YmyD = float(Yd)/1000.0-0.2
    print(YmyP, YmyI, YmyD)

    #Show the result in frames



    if Xval <= (SPX-5) or Xval >= (SPX+5):
        servoOutputX = pidX(Xval)
        servoX.value = servoOutputX + Xcenter
    else:
        servoX.value = Xcenter
    if Yval <= (SPY-5) or Yval >= (SPY+5):
        servoOutputY = pidY(Yval)
        servoY.value = servoOutputY + Ycenter
    else:
        servoY.value = Ycenter


    cv2.imshow('tracking',frame)
    pidX.tunings = (-XmyP,-XmyI,-XmyD)
    pidY.tunings = (-YmyP,-YmyI,-YmyD)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()
