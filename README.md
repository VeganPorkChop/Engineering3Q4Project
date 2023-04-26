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
## Reflection Questions:
* What is your project?
Our project is an auto-ball balencing plate. Using PID in and outputs we plan to be able to make a ball stop in the center of the plate without human interference.
* What components are connected to your Raspberry Pi?
There are very few components connected to our Pi, we have two servos, a camera module, and a batterypack connected to a switch.
* Explain (in detail) how your Code Prototype works
Our prototype is very simple. We combined all of the example code from above and put it all into one document. Our code initially creates 3 windows. One for each PID tuning and one for the camera. Then it creates slide bars so we can update the PID values real-time. Then it finds a ball on the camera, draws a dot on the center of it, and gives us its centerpoint. With that information we use our PID to get it as close to the center as possible. Then we find the slide bar values, update the PID and move the servos. Finally, the sequence starts over.

# FINAL CODE
# CAD
[Onshape Doc](https://cvilleschools.onshape.com/documents/2733d03459af870860d20d9e/w/29cc8494b29da55394609a40/e/f19878d97a4d184786ee9736)
