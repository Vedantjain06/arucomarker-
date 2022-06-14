import cv2 as cv
from math import atan2, sqrt

from cv2 import destroyAllWindows
from object_detector import *
import cv2
import numpy as np
import cv2.aruco as aruco
import os
 
 # Load the image
img = cv.imread("CVtask.jpg")

def boundary(img,pts): #draw the points in the image for shape detection
    pts=np.int32(pts).reshape(-1,2)
    img=cv2.drawContours(img,[pts],-1,(255,255,0),-3)
    return img


def Angle(img, p1, p2,):
       #not using this one
    p = list(p1)
    q = list(p2)
  ## calculating the angle for orientation
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
 
 
#calculating out the geoorientation of the boxes 
def geoorientation(pts, img): #not sure whether this is needed or not
  # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64) #(sz,2) dimensions
    for i in range(data_pts.shape[0]):# range till the lenght 
    data_pts[i,0] = pts[i,0,0]
    data_pts[i,1] = pts[i,0,1]
 
  #perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
 
  # store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
  ## [pca]
 
geoangle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
return geoangle
 
 
cv.imshow('Input Image', img)
 
# Convert image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
# Convert image to canny 
canny = cv.Canny(img,50,255)
# performing the edge cascading to count the contours for shape detection

contours, _ = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
 # retr_list to return all the contours
 # chain_approx_none is used instead of 'simple' because we want all the contours

for i, c in enumerate(contours):
 
  # calculate the area of each contour
  area = cv.contourArea(c)
 
  # ignore contours that are too small or too large
  if area < 3000 or 100000 < area:
    continue

   
  # Find the orientation of each shape
  geoorientation(c, img)

#colour detection
#converting the image into hsv image 

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


 # for green
    # define mask
greenlower = np.array([136, 87, 111], dtype='uint8')
greenupper = np.array([180, 255, 255], dtype='uint8')
greenmask = cv2.inRange(hsv, greenlower, greenupper) #inrange funtion to check the our hsv image array is between the defined array bounds 

greeninter=cv2.bitwise_and(img,img,mask=greenmask) #used the bitwise and operator with img and the greemmasked img to detect green colour

    #for orange
    # define mask

orangelower = np.array([136, 87, 111], dtype='uint8')
orangeupper = np.array([180, 255, 255],dtype='uint8')
orangemask = cv2.inRange(hsv,orangelower,orangeupper)
orangeinter=cv2.bitwise_and(img,img,mask=orangemask)

    #for black
    #define mask

blacklower = np.array([136, 87, 111],dtype='uint8')
blackupper = np.array([180, 255, 255],dtype='uint8')
blackmask = cv2.inRange(hsv,blacklower,blackupper)
blackinter=cv2.bitwise_and(img,img,mask=blackmask)


    #for pink-peach
    # define mask
pplower = np.array([136, 87, 111], dtype='uint8')
ppupper = np.array([180, 255, 255], dtype='uint8')
ppmask = cv2.inRange(hsv, pplower, ppupper)
ppinter=cv2.bitwise_and(img,img,mask=ppmask)

    #in colourinter variables the image with only the respective colour part of the image is stored 

    #now the last part of our code is to put the aruco marker on the boxes according to the rotation angle and colour

    #first we'll rorate and change the orentation of marker if requrired for
    #that we'll create a rotate funtion

def  rotate(image,(rotpoint=(image.shape(0)//2),(image.shape(1)//2)),angle()):#setting rotation point as centre
    (height,width) = (image.shape(1),image.shape(0))

    rotmat=cv.getRotationMatrix2D(rotpoint,angle1,1.0)#here angle1 i angle at which the image has to be rotated and 1 is scale value
    dimension=(width,height)
    return cv2.warpAffine(image,rotmat,dimension)    

#now according we can rotate each masked colour image by calling the angle from the angle funtioon
rotate(markerimage1,(),(),angle) #angle funtion will be called for the the angle of rotation
rotate(markerimage2,(),(),angle) #dimensions will be taken by default
rotate(markerimage3,(),(),angle)
rotate(markerimage4,(),(),angle)

#we will now agument the markers on our shapes 

dictionary=cv.aruco.Dictionary_get(cv.aruco.DICT_5X5_50)
markerimage1= cv.imread('ha.jpg')
markerimage2= cv.imread('HaHa.jpg')
markerimage3= cv.imread('LMAO.jpg')
markerimage4= cv.imread('XD.jpg')

if img==greeninter:
    greeninter=cv.drawMarker(markerimage1,0,250,int,50,1,)
elif img==orangeinter:
    orangeinter=cv.drawMarker(markerimage2,0,250,int,50,1,)
elif img==blackinter:
    blackinter=cv.drawMarker(markerimage3,0,250,int,50,1,)
elif img==ppinter:
    ppinter=cv.drawMarker(markerimage4,0,250,int,50,1,)

cv.waitkey(0)
destroyAllWindows()
        
        
        

