
# coding: utf-8

# In[ ]:


from pdf2image import convert_from_path
from PIL import Image
import PIL
import numpy as np
import cv2
from itertools import chain
import math
from skimage import transform
from xml.etree import ElementTree
import time

# In[2]:


def xml2dict(path):
    ea= ElementTree.parse(path).getroot()
    dicta={}
    rows= int(ea.find('size')[1].text)
    cols=int(ea.find('size')[0].text)
    for atype in ea.findall('.//object'):
        pt1=(int(atype[4][0].text),int(atype[4][1].text))   
        pt2=(int(atype[4][2].text),int(atype[4][3].text))
        dicta[atype[0].text]=tuple([pt1,pt2])
    return dicta,rows,cols


# In[ ]:


def rotatefine(dst_im):
    img2=dst_im.copy()
    gray=cv2.cvtColor(dst_im, cv2.COLOR_BGR2GRAY)
    thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,20)
    blurred = cv2.GaussianBlur(thresh, (3, 3), 0)
    edges = cv2.Canny(blurred, 150, 500, apertureSize=3)
    #if (debug_mode):  show_image(edges, window_name)

    lines= cv2.HoughLines(edges, 1, math.pi/180.0, 300, np.array([]), 0, 0)
    rows,cols,_=img2.shape
    #lines1 = lines1[0]
    #lines2 = lines2[0]
    ang=[]
    a,b,c = lines.shape
    for i in range(a):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        if int(math.degrees(theta)) in range(0,5) or int(math.degrees(theta)) in range(175,180):
            ang.append(math.degrees(theta))
            a = math.cos(theta)
            b = math.sin(theta)
            x0, y0 = a*rho, b*rho
            pt1 = ( int(x0+cols*(-b)), int(y0+rows*(a)) )
            pt2 = ( int(x0-cols*(-b)), int(y0-rows*(a)) )
            cv2.line(img2, pt1, pt2, (0, 0, 255), 5, cv2.LINE_AA)
    angle= max(set(ang), key=ang.count)
    if angle>90:
        angle=angle-180.0
    # angle=min(angle,180.0-angle)
#     print angle


    if (debug_mode):  show_image(img2, window_name)
    rot=transform.rotate(dst_im,angle,mode='edge')
    rot=np.multiply(rot,255.0)
    rot=rot.astype(np.uint8)
#     print rot.shape
    if (debug_mode):  show_image(rot, window_name)
    return rot


# In[ ]:


def get_image_width_height(image):
    image_width = image.shape[1]  # current image's width
    image_height = image.shape[0]  # current image's height
    return image_width, image_height
def show_image(image, window_name):
    # Show image
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name,1200,1200)
#     cv2.startWindowThread()
    cv2.imshow(window_name, image)
    image_width, image_height = get_image_width_height(image)
#     cv2.resizeWindow(window_name, image_width, image_height)

    # Wait before closing
    cv2.waitKey(0) & 0xFF
    

    cv2.destroyAllWindows()


# In[5]:


def detect_box(image, cropIt=True):

    im=image.copy()
    image_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    image_y = np.zeros(image_yuv.shape[0:2], np.uint8)
    image_y[:, :] = image_yuv[:, :, 0]
    thr=min(image_y[0][0],image_y[-1][-1])
    image_blurred = cv2.GaussianBlur(image_y, (3, 3), 0)
    #if (debug_mode):  show_image(image_blurred, window_name)
    edges = cv2.Canny(image_blurred, 100, 300, apertureSize=3)

    #if (debug_mode): show_image(edges, window_name)


    _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours, key = cv2.contourArea, reverse = True)
    cnt=contours[0:5]
    if (debug_mode):
         cv2.drawContours(im, cnt, -1, (0, 255, 0), 3)
         #show_image(im, window_name)

    allbox=[]
    bstcnt=[]
    best_box = [-1, -1, -1, -1]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if best_box[0] < 0:
            best_box = [x, y, x + w, y + h]

        else:
            if x < best_box[0]:
                best_box[0] = x
            if y < best_box[1]:
                best_box[1] = y
            if x + w > best_box[2]:
                best_box[2] = x + w   
            if y + h > best_box[3]:
                best_box[3] = y + h
        best_box=[best_box[0],best_box[1],best_box[2],best_box[3]]  
        allbox.append(best_box)

    cv2.rectangle(im, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (0, 255, 0), 1)
    if (debug_mode):show_image(im, window_name)

    if (cropIt):
        image = image[best_box[1]:best_box[3], best_box[0]:best_box[2]]
        if (debug_mode): show_image(image, window_name)

    return image


# In[ ]:


def threshold(image):
    rot1=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
# rot1=cv2.adaptiveThreshold(rot1,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,5)
    rot1[rot1>200] = 255
    rot1[rot1<190]=0
#     if (debug_mode): show_image(rot1, window_name)
    return rot1


# In[ ]:


def tempmatch(image,temp,thresh,crop):
    im=image.copy()
    h, w = temp.shape
    res = cv2.matchTemplate(image,temp,cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    rows,cols=image.shape
    if top_left[1]>rows/2:
        M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
        im = cv2.warpAffine(im,M,(cols,rows))
        image = cv2.warpAffine(image,M,(cols,rows))
        thresh=cv2.warpAffine(thresh,M,(cols,rows))
        crop=cv2.warpAffine(crop,M,(cols,rows))
        res = cv2.matchTemplate(im,temp,cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(image,top_left, bottom_right, (0,0,255), 10)
    if (debug_mode): show_image(image, window_name)
    return crop,thresh,im,tuple([top_left,bottom_right]),max_val


# In[ ]:


if __name__ =='__main__':
    import os,sys
    import argparse
    from pdf2image import convert_from_path
    from PIL import Image
    import PIL
    import numpy as np
    import cv2
    from itertools import chain
    import math
    from skimage import transform
    from xml.etree import ElementTree
    from pytesseract import image_to_string
    import pytesseract
    import json

    window_name = 'window'
#     size_max_image = 500
    script_dir=os.path.dirname(__file__)
    debug_mode=False
    parser=argparse.ArgumentParser()
    parser.add_argument('-f','--file',help='address to file')
    parser.add_argument("-d","--debug", help="specify whether to show steps",
                    action="store_true")
    parser.add_argument("-o","--output", help="where to store output",
                    default=os.path.join(script_dir,'./'))
    parser.add_argument('-l','--list',nargs='+')

    args=parser.parse_args()

<<<<<<< HEAD
    folder_to_check = args.file
=======
    folder_to_check = "/home/entrophy/Downloads/crm_highlighter/10/"
>>>>>>> 067f6a6cc1e58777fea791631815775b4e7a79ef
    # output_csv_file = '/home/deepmind/abs/samples/generated_data/shivam_good_data/good_data_300dpi/ssd_test_output_all_40.csv'

    fileList = []
    for file in os.listdir(folder_to_check):
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".JPG") or file.endswith(".JPEG"):
            full_file_name = os.path.join(folder_to_check, file)
            #print (full_file_name)
            fileList.append(full_file_name)

    #new_file_list = fileList[2516:]

    for eachFile in fileList:
        filepath=eachFile
        ext=os.path.splitext(filepath)[1]
        image=[]
        image.append(cv2.imread(filepath))
        if (args.debug):
            debug_mode=True
        
        outpath=args.output
        highlightlist=args.list
        template_path='./template.jpg'
        alabels='./crfformpg1.xml'
        blabels='./crfformpg2.xml'
        dicta,rows,cols=xml2dict(os.path.join(script_dir,alabels))
        dictb,_,_=xml2dict(os.path.join(script_dir,blabels))
        temp=cv2.imread(os.path.join(script_dir,template_path))
        pytesseract.pytesseract.tesseract_cmd='/usr/bin/tesseract'
        for idx,im in enumerate(image):
            if (debug_mode): show_image(im, window_name)
            rot=rotatefine(im)
            crop=detect_box(rot)
            crop1=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
            ret,thresh=cv2.threshold(crop1,200, 255, 0)
            thr=threshold(crop)
            newtemp=threshold(temp)
            fx=float(thr.shape[1])/cols
            fy=float(thr.shape[0])/rows
            newtemp=cv2.resize(newtemp,(int(fx*temp.shape[1]),int(fy*temp.shape[0])),cv2.INTER_AREA)
            crop,thresh,ims,logopos,maxval=tempmatch(thr,newtemp,thresh,crop)
            logopos_act=dicta['temp']
            hrat=float(np.sqrt(((logopos[0][0]- logopos[1][0])**2)))/np.sqrt(((logopos_act[0][0]- logopos_act[1][0])**2))
            vrat=float(np.sqrt(((logopos[0][1]- logopos[1][1])**2)))/np.sqrt(((logopos_act[0][1]- logopos_act[1][1])**2))
            log=thresh[logopos[0][1]:logopos[1][1],logopos[0][0]:logopos[1][0]]
            #show_image(log,window_name)
            hshift=int(((logopos[0][0]-logopos_act[0][0]*hrat)+(logopos[1][0]-logopos_act[1][0]*hrat))/2)
            vshift=int(((logopos[0][1]-logopos_act[0][1]*vrat)+(logopos[1][1]-logopos_act[1][1]*vrat))/2)
            #print logopos,tuple([(int(x[0]*hrat),int(x[1]*vrat)) for x in logopos_act])    
            #print hshift,vshift
            formname=tuple([(int(x[0]*hrat),int(x[1]*vrat+vshift)) for x in dicta['form name']])

            newthr=ims[formname[0][1]:formname[1][1],formname[0][0]:formname[1][0]]
            #cv2.imwrite('/home/nishant/Desktop'+'/'+os.path.splitext(os.path.basename(filepath))[0]+str(np.random.randint(200))+'.jpg',newthr)

            #Image.fromarray(newthr).show()
            #  formname= image_to_string(Image.fromarray(thr[formname[0][1]:formname[1][1],formname[0][0]:formname[1][0]]))
            formname1= image_to_string(Image.fromarray(newthr),config='--psm=13')
            for j in reversed(formname1):
                if j=='A':
                    dictref=dicta
                    break
                elif j=='B':
                    dictref=dictb
                    break
                else:
                    dictref=None
            if dictref is None:
                print('hola')
                kernel = np.ones((3,3),np.uint8)
                newthr=thresh[formname[0][1]:formname[1][1],formname[0][0]:formname[1][0]]
                newthr=cv2.morphologyEx(newthr, cv2.MORPH_OPEN, kernel)
                formname1= image_to_string(Image.fromarray(newthr),config='--psm=13')
                for j in reversed(formname1):
                            if j=='A':
                                dictref=dicta
                                break
                            elif j=='B':
                                dictref=dictb
                                break
            dictnew={}
            if dictref:
                for key in dictref:         
                    dictnew[key]=tuple([(min(thr.shape[1],int(x[0]*hrat+hshift)),int(min(thr.shape[0],x[1]*vrat+vshift))) if x[0]<int(thr.shape[1]/2) else (min(thr.shape[1],int(x[0]*hrat)),int(min(thr.shape[0],x[1]*vrat+vshift))) for x in dictref[key]])
            else:
                print("ocr didn't work")
                sys.exit
            to_store=[dictnew,crop]

            for key, value in dictnew.items():
                print ("key-", key, ",value-",value)
                crop_demo = crop
                print ("dictionary", dictnew[key][0])

                x1, y1 = dictnew[key][0]
                x2, y2 = dictnew[key][1]

                source = crop_demo.copy()


                # crop_show = cv2.rectangle(crop_demo.copy(),dictnew[key][0],dictnew[key][1],(0,255,0),5)
                crop_show = cv2.rectangle(crop_demo.copy(),(x1-20, y1-20), (x2+20, y2+20),(0,0,255),20)
                crop_show = cv2.addWeighted(crop_show, 0.5, source, 1 - 0.5, 0, source)

                font = cv2.FONT_HERSHEY_SIMPLEX
                crop_show = cv2.putText(crop_show,key,(10,500), font, 3,(255,0,0),2,cv2.LINE_AA)
                # k=cv2.waitKey(0) & 0xFF
                # if k==ord('q'):
                #     break
                show_image(crop_show,window_name)
                    
                    






