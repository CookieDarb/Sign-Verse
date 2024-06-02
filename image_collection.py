#left right swapped
import cv2
from cvzone.HandTrackingModule import HandDetector
from turtle import delay 
import numpy as np
import math
import time
cap = cv2.VideoCapture(0)
img_size = 300
offset =20
detector = HandDetector(maxHands=2)
counter =0
rcnt=0
lcnt=0
folder = "C:\\ISL_recognition\\data\\O"
while True :
    success, img =cap.read()
    hands ,img = detector.findHands(img)
    if hands:
        imgWhite = np.ones((300, 600, 3), np.uint8) * 255
        for i in hands:
            if i['type']=='Left':
                lcnt = 1
                left = i
                xl, yl, wl, hl = left['bbox']
                imgCropl = img[yl - offset: yl + hl + offset, xl - offset+10: xl + wl + offset]
                imgCropshape = imgCropl.shape

                ar = hl / wl  # aspect ratio
                if ar > 1:
                    k = img_size / hl
                    wcal = math.ceil(wl * k)
                    imgResize = cv2.resize(imgCropl, (wcal, img_size))
                    wgap = math.ceil((img_size - wcal) / 2)
                    imgWhite[:300, 300+wgap:300+wgap + wcal] = imgResize
                else:
                    k = img_size / wl
                    hcal = math.ceil(hl * k)
                    imgResize = cv2.resize(imgCropl, (img_size, hcal))
                    hgap: int = math.ceil((img_size - hcal) / 2)
                    imgWhite[0+hgap:hgap + hcal, 300:600] = imgResize
                cv2.imshow("imageCrop", imgCropl)

            if i['type']=='Right':
                rcnt =1
                right = i
                xr, yr, wr, hr = right['bbox']
                imgCropr = img[yr - offset: yr + hr + offset, xr - offset: xr + wr + offset-10]
                imgCropshape = imgCropr.shape

                ar = hr / wr  # aspect ratio
                if ar > 1:
                    k = img_size / hr
                    wcal = math.ceil(wr * k)
                    imgResize = cv2.resize(imgCropr, (wcal, img_size))
                    wgap = math.ceil((img_size - wcal) / 2)
                    imgWhite[:, wgap:wgap + wcal] = imgResize
                else:
                    k = img_size / wr
                    hcal = math.ceil(hr * k)
                    imgResize = cv2.resize(imgCropr, (img_size, hcal))
                    hgap = math.ceil((img_size - hcal) / 2)
                    imgWhite[hgap:hgap + hcal, :300] = imgResize
                cv2.imshow("imageCrop", imgCropr)



        cv2.imshow("imageWhite", imgWhite)

    cv2.imshow("Webcam", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #
    key = cv2.waitKey(1)
    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
