# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 21:40:27 2025

@author: emirh
"""

import cv2 
import numpy as np 
from collections import deque

#nesne merkezini depolayacak veri tipi
buffer_size = 16 
pts= deque(maxlen = buffer_size)


# mavi renk aralığı HSV
blueLower = (90, 50, 50)   # Alt sınır (Hue, Saturation, Value)
blueUpper = (130, 255, 255)  # Üst sınır
# kırmızı renk arlığı HSV
lower_red1 = (0, 120, 70)
upper_red1 = (10, 255, 255)
lower_red2 = (170, 120, 70)
upper_red2 = (180, 255, 255)


# capture
cap = cv2.VideoCapture(0)
cap.set(3,960)
cap.set(4,480)

while True:
    
    success , imgOriginal = cap.read()
    
    if success:
        
        # blur
        blurred = cv2.GaussianBlur(imgOriginal , (11,11), 0)
        
        # hsv
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
#        cv2.imshow("HSV Image",hsv)
        
        # mavi için maske oluştur
        maskBlue = cv2.inRange(hsv, blueLower, blueUpper)
#        cv2.imshow("Mask Image",mask)
        
        # kırmızı için maske oluştur
        maskRed1 = cv2.inRange(hsv, lower_red1, upper_red1)
        maskRed2 = cv2.inRange(hsv, lower_red2, upper_red2)
        maskRed = cv2.bitwise_or(maskRed1, maskRed2)
        
        
        
        
        # maskblue nin etrafında kalan görüntüleri sil
        maskBlue = cv2.erode(maskBlue, None, iterations = 2)
        maskBlue = cv2.dilate(maskBlue, None, iterations= 2)
        cv2.imshow("MaskBlue + erozyon ve genisletme", maskBlue)
        
        
        # maskred nin etrafında kalan görüntüleri sil
        maskRed = cv2.erode(maskRed, None, iterations = 2)
        maskRed = cv2.dilate(maskRed, None, iterations= 2)
        cv2.imshow("MaskRed + erozyon ve genisletme", maskRed)
        
        # kontur blue
        contoursBlue, _ = cv2.findContours(maskBlue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # kontur red
        contoursRed, _ = cv2.findContours(maskRed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        center = None
        rectBlue = None
        rectRed = None
        
        if len(contoursBlue) > 0:
            cBlue = max(contoursBlue, key=cv2.contourArea)
            rectBlue = cv2.minAreaRect(cBlue)
            boxBlue = cv2.boxPoints(rectBlue)
            boxBlue = np.int64(boxBlue)
        
            MBlue = cv2.moments(cBlue)
            centerBlue = (int(MBlue["m10"]/MBlue["m00"]), int(MBlue["m01"]/MBlue["m00"]))
        
            cv2.drawContours(imgOriginal, [boxBlue], 0, (0,255,255), 2)
            cv2.circle(imgOriginal, centerBlue, 5, (255,0,255), -1)
        
        if len(contoursRed) > 0: 
            cRed = max(contoursRed, key=cv2.contourArea)
            rectRed = cv2.minAreaRect(cRed)
            boxRed = cv2.boxPoints(rectRed)
            boxRed = np.int64(boxRed)
        
            MRed = cv2.moments(cRed)
            centerRed = (int(MRed["m10"]/MRed["m00"]), int(MRed["m01"]/MRed["m00"]))
        
            cv2.drawContours(imgOriginal, [boxRed], 0, (0,255,255), 2)
            cv2.circle(imgOriginal, centerRed, 5, (255,0,255), -1)

            
            '''            
            ((x,y), (width,height), rotation) = rect
            
            s = "x: {}, y: {}, width: {}, height : {}, rotation: {}".format(np.round(x),np.round(y),np.round(width),np.round(height),np.round(rotation))  
            
            print(s)
            '''            

            # kutucuk
            boxBlue = cv2.boxPoints(rectBlue)
            boxBlue = np.int64(boxBlue)
            
            boxRed = cv2.boxPoints(rectRed)
            boxRed = np.int64(boxRed)
            
            
            # moment
            MBlue = cv2.moments(cBlue)
            centerBlue = (int(MBlue["m10"]/MBlue["m00"]), int(MBlue["m01"]/MBlue["m00"]))
            
            MRed = cv2.moments(cRed)
            centerRed = (int(MRed["m10"]/MRed["m00"]), int(MRed["m01"]/MRed["m00"]))
            
            
            # konturu cizdir: sarı
            cv2.drawContours(imgOriginal, [boxBlue], 0, (0,255,255), 2)
            cv2.drawContours(imgOriginal, [boxRed], 0, (0,255,255), 2)
            
            # merkeze bir tane nokta çizelim: 
            cv2.circle(imgOriginal, centerBlue, 5,(255,0,255),-1)
            cv2.circle(imgOriginal, centerRed, 5,(255,0,255),-1)
            '''
            # bilgileri ekrana yazdır
            cv2.putText(imgOriginal, s, (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1 , (255,255,255), 2 )
            '''
        imgBlue = imgOriginal.copy()
        imgRed = imgOriginal.copy()

        cv2.imshow("Blue Detection", imgBlue)
        cv2.imshow("Red Detection", imgRed)
        
        
        
    if cv2.waitKey(1) == ord("q") & 0xFF: 
        break
            
cap.release()
cv2.destroyAllWindows()            