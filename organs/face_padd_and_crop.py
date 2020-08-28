'''
This script takes in 4 arguments(face detected image , image_type, rectangle, padding constant)

image_type = "cv2" or "pil"

1. Tries to increase the region of interest (ROI) of the rectangle by specified constant.
2. Suppose the ROI crosses the borders of the input image then it restricts the padding
to the possible maximum
3. Draw borders around image
4. returns the increased ROI image


Pending : have to write for pil image if required

'''

import cv2
import numpy as np
from PIL import Image
import os
import sys

pathToScriptFolder = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(pathToScriptFolder, '../'))

from organs import image_formats as frmt


def face_padd_and_crop(source_image, image_type, rec, cnst):

    if image_type == "cv2":
        h_source_image ,w_source_image, c_source_image = source_image.shape
        top, right, bottom, left = rec

        def check_rec(top, right, bottom, left):
            rec_status = False
            if top < bottom and left < right:
                rec_status = True
            return rec_status


        def check_cnst(top, right, bottom, left, cnst):
            cnst_status = False
            if top-cnst >=0 and bottom+cnst <= h_source_image and left-cnst >=0 and right+cnst <= w_source_image:
                cnst_status = True
            return cnst_status


        if check_rec(top, right, bottom, left):
            if check_cnst(top, right, bottom, left, cnst):
                face = source_image[top-cnst : bottom+cnst, left-cnst : right+cnst]
                return face
            else:
                for i in range(0,10000) :
                    # print(i)
                    if cnst - i*1 > 0:
                        new_cnst = cnst - i*1
                        if check_cnst(top, right, bottom, left, new_cnst):
                            face = source_image[top-new_cnst : bottom+new_cnst, left-new_cnst : right+new_cnst]
                            return face
                            break
                        else:
                            # print ("new constant did not work", str(new_cnst))
                            pass
                # print(" padded image was out of border but corrected to max")
        else:
            # print ("input rec is incorrect, returing source image")
            return source_image


if __name__ == "__main__":

    image = cv2.imread("/home/yuyudhan/abs/qualification_project/imgs/form1page____4ae.jpg")
    rec = [100,300,200,200]
    #rec = [1800, 366, 1900, 200]
    face = face_padd_and_crop(image, "cv2", rec, 0)
    # cv2.imshow("paddedFace", face)
    # cv2.waitKey(0)
    # cv2.destroyAllWindow
