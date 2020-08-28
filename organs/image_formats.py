'''
This module converts the images from one format to another.
Eg convert PIL format to opencv format

'''

from PIL import Image
import cv2
import numpy as np

def convertOpenCvToPIL(cv2_im):
	cv2_im = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
	pil_im = Image.fromarray(cv2_im)
	return pil_im
	pass

def convertPILToOpenCv(pil_im):
	# pil_im = PIL.Image.open(pil_im).convert('RGB')
	cv2_im = np.array(pil_im)
	# Convert RGB to BGR
	cv2_im = cv2_im[:, :, ::-1].copy()
	return cv2_im
	pass

def convertImageFormat(image, fromFormat, toFormat):
	# Python does not have switch case syntax
	if fromFormat == 'cv2' and toFormat == 'pil':
		return convertOpenCvToPIL(image)
		pass
	elif fromFormat == 'pil' and toFormat == 'cv2':
		return convertPILToOpenCv(image)
		pass
	else:
		return "Couldn't convert"
		pass
	pass
