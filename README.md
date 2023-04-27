# PLANNING
## Checklist
* Create cad
* Create camera code
* Use camera to detect ball
* Create servo code
* Create PID code
* Combine all code
* Finish documentation
### Weekly Updates
* Week 4/24-31
 -Documentation just started and all the code functions individually except for the pid code. It, in concept, functions perfectly, but needs lots of tuning.
 -One problem we have is that the servos are off center, originally we thought about changing the set point of the PID function to accomidate, but instead of dealing with I we decided to level the servo instead and offset its degrees.
 
# CODE
## Camera Code + Detection
http://trevorappleton.blogspot.com/2013/11/python-getting-started-with-opencv.html
```py
import os
import cv2
import math

##Resize with resize command
def resizeImage(img):
    dst = cv2.resize(img,None, fx=0.25, fy=0.25, interpolation = cv2.INTER_LINEAR)
    return dst

##Take image with Raspberry Pi camera
os.system("raspistill -o image.jpg")

##Load image
img = cv2.imread("/home/pi/Desktop/image.jpg") 
grey = cv2.imread("/home/pi/Desktop/image.jpg",0) #0 for grayscale

##Run Threshold on image to make it black and white
ret, thresh = cv2.threshold(grey,50,255,cv2.THRESH_BINARY)

##Use houghcircles to determine centre of circle
circles = cv2.HoughCircles(thresh,cv2.cv.CV_HOUGH_GRADIENT,1,75,param1=50,param2=13,minRadius=0,maxRadius=175)
for i in circles[0,:]:
    #draw the outer circle
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    #draw the centre of the circle
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

##Determine co-ordinates for centre of circle
x1 = circles[0][0][0]
y1 = circles[0][0][1]
x2 = circles[0][1][0]
y2 = circles[0][1][1]
##Angle betwen two circles
theta = math.degrees(math.atan((y2-y1)/(x2-x1)))

##print information
print "x1 = ",x1
print "y1 = ",y1
print "x2 = ",x2
print "y2 = ",y2
print theta
print circles

##Resize image
img = resizeImage(img)
thresh = resizeImage(thresh)
##Show Images 
cv2.imshow("thresh",thresh)
cv2.imshow("img",img)

cv2.waitKey(0)
```
This code uses OpenCV to edge detect using Hue Saturation and Color, it combines the three and out pops your circle. The explination and how to install is all of the link.
## Servo Code
https://gist.github.com/elktros/384443b57a33f399a4acba76191e0e63
```py
import RPi.GPIO as GPIO
import time

control = [5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]

servo = 22

GPIO.setmode(GPIO.BOARD)

GPIO.setup(servo,GPIO.OUT)
# in servo motor,
# 1ms pulse for 0 degree (LEFT)
# 1.5ms pulse for 90 degree (MIDDLE)
# 2ms pulse for 180 degree (RIGHT)

# so for 50hz, one frequency is 20ms
# duty cycle for 0 degree = (1/20)*100 = 5%
# duty cycle for 90 degree = (1.5/20)*100 = 7.5%
# duty cycle for 180 degree = (2/20)*100 = 10%

p=GPIO.PWM(servo,50)# 50hz frequency

p.start(2.5)# starting duty cycle ( it set the servo to 0 degree )


try:
       while True:
           for x in range(11):
             p.ChangeDutyCycle(control[x])
             time.sleep(0.03)
             print x
           
           for x in range(9,0,-1):
             p.ChangeDutyCycle(control[x])
             time.sleep(0.03)
             print x
           
except KeyboardInterrupt:
    GPIO.cleanup()
```
Unlike C++, Python bases its arduino code off of hertz. That means that instead of 180* is -0.5 to 0.5.
## PID Code
https://pypi.org/project/simple-pid/
```py
from simple_pid import PID
pid = PID(1, 0.1, 0.05, setpoint=1)#IMPORTANT

# Assume we have a system we want to control in controlled_system
v = controlled_system.update(0)

while True:
    # Compute new output from the PID according to the systems current value
    control = pid(v)#IMPORTANT

    # Feed the PID output to the system and get its current value
    v = controlled_system.update(control)#IMPORTANT
```
This code is a non-functioning example, it just shows how you'd call a function. The main lines are as marked, I suggest looking them up in the link to better understand the code.
# CODE PROTOTYPE
```py
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
```
## Prototype Video
![ezgif com-video-to-gif (2)](https://user-images.githubusercontent.com/91289762/234966713-8d4e86ee-c05f-425c-882f-11245f7f868c.gif)
## Reflection Questions:
* **What is your project?**
Our project is an auto-ball balencing plate. Using PID in and outputs we plan to be able to make a ball stop in the center of the plate without human interference.
* **What components are connected to your Raspberry Pi?**
There are very few components connected to our Pi, we have two servos, a camera module, and a batterypack connected to a switch.
* **Explain (in detail) how your Code Prototype works**
Our prototype is very simple. We combined all of the example code from above and put it all into one document. Our code initially creates 3 windows. One for each PID tuning and one for the camera. Then it creates slide bars so we can update the PID values real-time. Then it finds a ball on the camera, draws a dot on the center of it, and gives us its centerpoint. With that information we use our PID to get it as close to the center as possible. Then we find the slide bar values, update the PID and move the servos. Finally, the sequence starts over.
* **What has been the hardest part?**
The hardest part so far would definatly have been getting used to python. For the past 3 years I've done all of my coding in C++. This is my first time coding in python
* **What have you learned along the way?**
I've learned that one major differnce between C++ and Python is the variable definition. Python always assumes its a integer, whereas C++ makes you specify that. I've also learned that both Python and C++ are esentially the same, but just with different functions.
* **What are your immediate next steps?**
As of 4/26/23 we need to redesign the base plate and make a more structurally sound building, also re-level the servos so that they're not implicitly trying to throw our pingpong ball away. The next would be to complete our project milestones.
# FINAL CODE
# CAD
[Onshape Doc](https://cvilleschools.onshape.com/documents/2733d03459af870860d20d9e/w/29cc8494b29da55394609a40/e/f19878d97a4d184786ee9736)
