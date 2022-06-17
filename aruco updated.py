from cv import rect
import numpy as np
import cv2 as cv
import cv2.aruco as aruco
import math




# defining the funtion for id information
#will tell the valid and invalid ids
#also this funtion will give info about corners
def findmarker(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    key =getattr(aruco,f'DICT_5X5_250')
    dict = aruco.Dictionary_get(key)
    dectparametre = aruco.DetectorParameters_create()
    (corners,ids,rejected) = cv.aruco.detectMarkers(img,dict,parameters =dectparametre)
    aruco.drawDetectedMarkers(img,corners)
    return corners,ids

#reading arucomarkers 
markerimage1= cv.imread('ha.jpg')
markerimage2= cv.imread('HaHa.jpg')
markerimage3= cv.imread('LMAO.jpg')
markerimage4= cv.imread('XD.jpg')


l=[markerimage1,markerimage2,markerimage3,markerimage4]
rotatedfinal = [0,0,0,0] #list of final rotated markers
for i in range(4):
    p,d = findmarker(l[i]) #calling the funtion to get 
    tl = (int(p[0][0][0][0]),int(p[0][0][0][1]))
    bl = (int(p[0][0][3][0]),int(p[0][0][3][1]))
    tr = (int(p[0][0][1][0]),int(p[0][0][1][1]))
    br = (int(p[0][0][2][0]),int(p[0][0][2][1]))
    cx = int((tl[0]+br[0])/2) #centre
    cy = int((tl[1]+br[1])/2) #centre
    tantheta = math.atan2(br[1]-bl[1])/(br[0]-bl[0])
    angle = math.degrees(tantheta) #to convert radian into degree
    rotmat =cv.getRotationMatrix2D((cx,cy),angle,1.0) 
    rotated = cv.warpAffine(l[i], rotmat, (300,300))
    p,q = findmarker(rotated)
    tl = (int(p[0][0][0][0]),int(p[0][0][0][1]))
    tr = (int(p[0][0][1][0]),int(p[0][0][1][1]))
    bl = (int(p[0][0][3][0]),int(p[0][0][3][1]))
    br = (int(p[0][0][2][0]),int(p[0][0][2][1]))
    
   # cropping arucomarkers 
    rotatedfinal[i] = rotated[tl[1]:bl[1],tl[0]:tr[0]]
   
    cv.imshow("img",rotatedfinal[i])
    cv.imshow("img_1",rotated)    
    cv.waitKey(0)  
    cv.destroyAllWindows()


#reading the image
image = cv.imread("Pictures//CVtask.jpg")
image =cv.resize(image,(0,0),fx = 0.5,fy = 0.5)
cv.imshow("shapes",image)
cv.waitKey(100)

 
# Convert image to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
 
# Convert image to canny 
canny = cv.Canny(image,50,255)
# performing the edge cascading to count the contours for shape detection

contours,hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
 # retr_list to return all the contours
 # chain_approx_none is used instead of 'simple' because we want all the contours


#detecting squares
 for c in contours: #cnts will be lenght of the contour list(cnts)
        app=cv.approxPolyDP(c,0.01*cv.arcLength(c,True),True)
        if len(app)==4:
            x1,y1,w,h=cv.boundingRect(app)
            
            rat = w/float(h)
            if  rat >= 0.97 and rat <= 1.03:
                rectangle=cv.minAreaRect(c)
                box=cv.boxPoints(rect)
        return box

               
       
        stl2 = (int(app[0,1]),int(app[0,0]))
        tr2= (int(app[1,1]),int(app[1,0]))
        br2 = (int(app[2,1]),int(app[2,0]))
        bl2= (int(app[3,1]),int(app[3,0]))
        lenght =int((int( math.sqrt((br2[1]-tl2[1])**2+(tr2[0]-tl2[0])**2))/2))
        tantheta2 = math.atan2((br2[0]-bl2[0])/(br2[1]-bl2[1]))
        angle2 = math.degree(tantheta2) 
    


# def warping(img,crns,id,image):
#     tl = crns[0][0],crns=[0][1]
#     tw = crns[1][0],crns=[1][1] #defining the corners 
#     bw = crns[2][0],crns=[2][1]
#     bl = crns[3][0],crns=[3][1]


        cv2.drawContours(image,[app],-1,(0,0,0),-1)
        blank =np.zeros([w,h,3],dtype='uint8') #defining a blank image of type int
            
        blank[int((w/2)-(1/2)):int((w/2)+(1/2)),int((h/2)-(1/2)):int((h/2)+(1/2))] = rotatedfinal[i]
        i+=1 #matching the aruco marker nd boundary rect centres
            
        rotblank=cv.getRotationMatrix2D((int(w/2),int(h/2)),-angle2,1.0)
        rotated = cv.warpAffine(blank,rotblank, (w,h))
           

            #using bitwise or operator for roatated and original image
        image[:2]=cv.bitwise_or(rotated,image[:2])
        cv.imshow("imagef",image)
cv.waitKey(0)
cv.destroyAllWindows()
