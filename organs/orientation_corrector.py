'''

This script takes in an input image and checks if it is rotated by 180 degrees

if so it corrects it back to zero degrees

Uses the triangles in the forms to do so

'''


from __future__ import division
import numpy as np
import argparse
import cv2
import imutils
import math
import uuid
import sys
import os
from scipy import ndimage

pathToScriptFolder = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(pathToScriptFolder, '../'))

from organs import geometry, image_manipulation

def correct_orientation_by_triangles(source_image_list):
    
    print ("###############################################")
    print ("orientation correction initiated....")
    print ("\n")

    corrected_image_list = []
    for each_image in source_image_list:

        img = imutils.resize(each_image, width=1637)
        height, width = img.shape[:2]

        # Checking horizontality of image
        # ######################################################
        if height < width:
            img = image_manipulation.rotate_bound(img, 90)
            pass

        # Image area detection for doing relative things later on
        # imgarea = height * width

        # ######################################################

        # By default setting uparmukhi
        orientation = "uparmukhi"

        # ========================
        # Working on the left block
        # ========================
        img = imutils.resize(each_image, width=1637)
        originalImage = img
        height, width = img.shape[:2]

        small_area = img[0:int(width/7), 0:int(height/10)]
        # small_area = img[0:int(height/10), (width - int(width/7)):width]

        image = small_area

        # find all the 'black' shapes in the image
        lower = np.array([0, 0, 0])
        upper = np.array([140, 140, 140])
        shapeMask = cv2.inRange(image, lower, upper)

        (cnts, _) = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL,
        	cv2.CHAIN_APPROX_SIMPLE)[-2:]
        # print "I found %d black shapes" % (len(cnts))
        # cv2.imwrite("tmp/mask-left.png", shapeMask)

        pageLeft = 0
        # loop over the contours
        for c in cnts:
            approx = cv2.approxPolyDP(c,0.1*cv2.arcLength(c,True),True)
            if len(approx) == 3:
                corners = [[approx[0][0][0], approx[0][0][1]], [approx[1][0][0], approx[1][0][1]], [approx[2][0][0], approx[2][0][1]]]
                anyoneZero = geometry.anyPointAboutZero(corners)
                area = geometry.polygonArea(corners)
                # print "***********"
                # print area
                # print anyoneZero
                if area > 130 and area < 250 and anyoneZero == False:
                    orientation = geometry.getTriangleOrientation(corners)
                    # print orientation
                    # print "$$$"
                    pageLeft = pageLeft + 1
                    # Contour drawing stopped to avoid test data desctruction
                    cv2.drawContours(image, [c], -1, (0, 0, 255), -1)


        # =========================
        # Working on left block completed
        # =========================


        # =========================

        # Working on the right block
        # ========================
        img = imutils.resize(each_image, width=1637)
        height, width = img.shape[:2]

        # small_area = img[0:int(width/7), 0:int(height/10)]
        small_area = img[0:int(height/10), (width - int(width/7)):width]

        image = small_area

        # find all the 'black' shapes in the image
        lower = np.array([0, 0, 0])
        upper = np.array([140, 140, 140])
        shapeMask = cv2.inRange(image, lower, upper)

        (cnts, _) = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL,
        	cv2.CHAIN_APPROX_SIMPLE)[-2:]
        # print "I found %d black shapes" % (len(cnts))
        # cv2.imwrite("tmp/mask-right.png", shapeMask)

        pageRight = 0
        # loop over the contours

        for c in cnts:
            approx = cv2.approxPolyDP(c,0.1*cv2.arcLength(c,True),True)
            if len(approx) == 3:
                corners = [[approx[0][0][0], approx[0][0][1]], [approx[1][0][0], approx[1][0][1]], [approx[2][0][0], approx[2][0][1]]]
                anyoneZero = geometry.anyPointAboutZero(corners)
                area = geometry.polygonArea(corners)
                # print "***********"
                # print area
                # print anyoneZero
                if area > 130 and area < 250 and anyoneZero == False:
                    orientation = geometry.getTriangleOrientation(corners)
                    # print orientation
                    # print "###"
                    pageRight = pageRight + 1
                    # Contour drawing stopped to avoid test data desctruction
                    cv2.drawContours(image, [c], -1, (0, 0, 255), -1)


        # =========================
        # Working on right block completed
        # =========================
     

        if pageRight > 4:
            orientation = "down"
            pass


        if orientation == 'down':
            print ("nichemukhi hai seedha kar raha hoon, will save the seedha image")
            pageRight, pageLeft = pageLeft, pageRight
            (h, w) = originalImage.shape[:2]
            center = (w / 2, h / 2)

            # rotate the image by 180 degrees
            M = cv2.getRotationMatrix2D(center, 180, 1.0)
            each_image = cv2.warpAffine(originalImage, M, (w, h))
            
        print (orientation)
        corrected_image_list.append(each_image)

    print("orientation correction completed")
    print ("###############################################")
    return corrected_image_list


if __name__ == "__main__":

    image_list = [
    cv2.imread("/home/signzy-engine/abs/ckyc-worker/data_files/form_images/cis_2.jpg"),
    cv2.imread("/home/signzy-engine/abs/ckyc-worker/data_files/form_images/cis_1.jpg")
    ]
    corrected = correct_orientation_by_triangles(image_list)
    # cv2.imshow("corrected", corrected[0])
    # cv2.imshow("corrected1", corrected[1])
    cv2.waitKey(0)




'''
def alignToNineties(image):
	if type(image) is str:
		image = cv2.imread(image)
		pass

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bitwise_not(gray)
	thresh = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	coords = np.column_stack(np.where(thresh > 0))
	angle = cv2.minAreaRect(coords)[-1]
	print angle
	angle = -angle

	(h, w) = image.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	rotated = cv2.warpAffine(image, M, (w, h),
		flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

	return rotated	
	pass

def removeUnnecessaryArea(image):
	if type(image) is str:
		image = cv2.imread(image)
		pass

	height, width = image.shape[:2]
	imageArea = height * width

	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	blurred = cv2.medianBlur(gray, 3)

	_,thresh = cv2.threshold(blurred,128,255,cv2.THRESH_BINARY)

	_, contours,h = cv2.findContours(thresh,1,2)

	largestAreaTillNow = cv2.contourArea(contours[0])
	largestContourTillNow = contours[0]

	for cnt in contours:
		thisContourArea = cv2.contourArea(cnt)
		if thisContourArea > 0:
			areaRatio = imageArea/thisContourArea
			pass
		else:
			areaRatio = 1
			pass
		if (0.98 < areaRatio < 1.02) == False:
			# cv2.drawContours(image,[cnt],0,255,5)
			if thisContourArea > largestAreaTillNow:
				largestAreaTillNow = thisContourArea
				largestContourTillNow = cnt
				pass
			pass
		pass

	x,y,w,h = cv2.boundingRect(largestContourTillNow)
	crop = image[y:y+h,x:x+w]
	# cv2.imshow("crop", crop)
	return crop

	pass

aligned = alignToNineties("/home/yuyudhan/abs/qualification_project/imgs/form1page____4ar.jpg")

cv2.imshow("aligned", aligned)

# borderRemoved = removeUnnecessaryArea(aligned)

# cv2.imshow("borderRemoved", borderRemoved)

cv2.waitKey(0)'''