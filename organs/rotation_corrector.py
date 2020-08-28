# -*- coding: utf-8 -*-
"""
Automatically detect rotation image of text using Radon transform or pyimagesearch text skew correction
If image is rotated by the inverse of the output, the lines will be
horizontal (though they may be upside-down depending on the original image)
It doesn't work with black borders
"""

from __future__ import division, print_function
from skimage.transform import radon
from PIL import Image
from numpy import asarray, mean, array, blackman
import numpy
from numpy.fft import rfft
import matplotlib.pyplot as plt
from matplotlib.mlab import rms_flat
import cv2
import os
import sys
import imutils
from scipy import ndimage
import numpy as np
import argparse
pathToScriptFolder = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(pathToScriptFolder, '../'))
from organs import image_formats as frmt


try:
    from parabolic import parabolic
    def argmax(x):
        return parabolic(x, numpy.argmax(x))[0]
except ImportError:
    from numpy import argmax


def find_rotation_with_radon(source_image):
	print ("calculating rotation with radon.....")

	source_image = imutils.resize(source_image, height = 500)
	pilImage =  frmt.convertImageFormat(source_image, "cv2", "pil")

	# converting to grayscale
	I = asarray(pilImage.convert('L'))
	I = I - mean(I)  # Demean; make the brightness extend above and below zero

	# Do the radon transform and display the result
	sinogram = radon(I)

	# Find the RMS value of each row and find "busiest" rotation,
	# where the transform is lined up perfectly with the alternating dark
	# text and white lines
	r = array([rms_flat(line) for line in sinogram.transpose()])
	rotation = argmax(r)
	print('rotation with radon: {:.2f} degrees'.format(rotation))
	return rotation

def find_rotation_with_column_stack(source_image):

	gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bitwise_not(gray)
	thresh = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	coords = np.column_stack(np.where(thresh > 0))
	angle = cv2.minAreaRect(coords)[-1]

	print('rotation with column stack: {:.2f} degrees'.format(90 - angle))
	return angle	
	pass


def rotate_image(source_image,angle):
	rotated_image = imutils.rotate_bound(source_image, 270+angle)
	return rotated_image


def find_and_rotate(source_image_list):
	corrected_source_image_list = []
	for each_image in source_image_list:
		angle = find_rotation_with_radon(each_image)
		corrected_image = rotate_image(each_image,angle)
		corrected_source_image_list.append(corrected_image)

	return corrected_source_image_list


if __name__ == "__main__":
	image_list = [
    cv2.imread("/home/yuyudhan/abs/signzySampleImages/aadhaar-basic/9ca16a7f30c981fe892ab4ab9dc093734390bfb9803379f4d0c470d8c6443ab2.jpeg"),
    cv2.imread("/home/yuyudhan/abs/qualification_project/imgs/form1page____4ar.jpg")
    ]
	images = find_and_rotate(image_list)
	cv2.imshow("aligned", images[0])
	cv2.imshow("aligned1", images[1])
	cv2.waitKey(0)