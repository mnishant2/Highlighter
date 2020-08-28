
'''
The idea of this module is that it will crop the a defined part of image from a parent image
image can assume two values
image = numpy representation of image obtained using cv2.imread() command
image = path on local disk where the image is stored.

Along with image it also takes the dimensions to crop

There may be some cases where the dimension maybe such that cropped image may

In that case if largestPossible is set to true then the largets possible image
starting from centroid is cropped. If centroid is also outside then error is returned.

If largestPossible is set to false (which is by default) then the module returns error if
cropping is attempted out of the original image

'''

def crop(image, dimensions, largestPossible):
	# Check if path or cv2.imread (numpy representation)

	# If image is a string check if the image exists at the mentioned location and
	# If does not exist then return error
	# If exists set image = cv2.imread(image)

	# If cv2.imread check if everything correct to be handled
	# Else return an error

	# This function in the end returns a cv2.imread (numpy) representation of the cropped image.
	pass
