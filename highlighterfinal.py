import os,sys
sys.path.append('./')
try:
	import argparse
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
	import imutils
	from organs import geometry, image_manipulation
except ImportError as ErrorMessage:
	print(ErrorMessage)
	sys.exit()

def triangle(img,shpMskValue=140):
    bot=False
    orientation=''
    count=0
    # try:
    lower = np.array([0, 0, 0])
    upper = np.array([shpMskValue, shpMskValue, shpMskValue])
    shapeMask = cv2.inRange(img, lower, upper)
    # except Exception as e:
    #     print("Triangle upper form classification left side slice error")
    im2, cnts, hierarchy = cv2.findContours(shapeMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        approx = cv2.approxPolyDP(c,0.03*cv2.arcLength(c,True),True)
        if len(approx) == 3:
            # print ("count")
            corners = [[approx[0][0][0], approx[0][0][1]], [approx[1][0][0], approx[1][0][1]], [approx[2][0][0], approx[2][0][1]]]
            anyoneZero = geometry.anyPointAboutZero(corners)
            area = geometry.polygonArea(corners)
            # print (area)
            if area > 100 and area < 260 and anyoneZero == False :
                l1 = geometry.distance(corners[0],corners[1])
                l2 = geometry.distance(corners[2],corners[1])
                l3 = geometry.distance(corners[0],corners[2])
                left_length_list = [l2/l1, l3/l1, l3/l2]
                count+=1
    #             print (left_length_list)
                left_length_status=True
                bot = True
                for length_ratio in left_length_list:
                    if length_ratio < 0.5 or length_ratio > 1.5:
                        left_length_status = False
                        break
                if left_length_status == True:
                    orientation = geometry.getTriangleOrientation(corners)
                    # print(orientation,count)
    return bot,count,orientation
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


def rotatefine(dst_im):
	
	img2=dst_im.copy()

	# print('No Image or Wrong Image')
 #        sys.exit()
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
	cv2.waitKey(0)
	cv2.destroyAllWindows()


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


def threshold(image):
	rot1=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	# rot1=cv2.adaptiveThreshold(rot1,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,5)
	rot1[rot1>200] = 255
	rot1[rot1<190]=0
	#     if (debug_mode): show_image(rot1, window_name)
	return rot1

def tempmatch(image,temp,thresh,crop,rot_check):
	im=image.copy()
	h, w = temp.shape
	res = cv2.matchTemplate(image,temp,cv2.TM_CCORR_NORMED)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	top_left = max_loc
	rows,cols=image.shape
	if not rot_check:
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


if __name__ =='__main__':
	import os,sys
	sys.path.append('./')
	try:
		import argparse
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
		from organs import geometry, image_manipulation
	except ImportError as ErrorMessage:
		print(ErrorMessage)
		sys.exit()
	window_name = 'window'
	dictref=None
	rot_check=False
	#     size_max_image = 500
	script_dir=os.path.dirname(__file__)
	debug_mode=False
	parser=argparse.ArgumentParser()
	parser.add_argument('-f','--file',help='address to file')
	parser.add_argument("-d","--debug", help="specify whether to show steps",
					action="store_true")
	parser.add_argument("-p","--part",type=str,help="specify part of the form")
	parser.add_argument("-o","--output", help="where to store output",
					default=os.path.join(script_dir,'./'))
	parser.add_argument('-l','--list',nargs='+')

	args=parser.parse_args()
	filepath=args.file
	ext=os.path.splitext(filepath)[1]
	im=cv2.imread(filepath)
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
	# pytesseract.pytesseract.tesseract_cmd='/usr/bin/tesseract'
	
	try:
		rot=rotatefine(im)
	except AttributeError:
		print('No Image or Wrong Image')
		sys.exit()
	if (debug_mode): show_image(im, window_name)
	# crop=detect_box(rot)
	crop=rot
	r,c=crop.shape[:2]
	if args.part and args.part=='A':
		dictref=dicta
	elif args.part and args.part=='B':
		dictref=dictb
	else:
		pass
	img = imutils.resize(im, width=1637)
	height, width = img.shape[:2]
	if height < width:
	    img = image_manipulation.rotate_bound(img, 90)
	    pass
	small_area_up=img[0:int(2*height/5),0:int(width/7)]
	small_area_bottom = img[int(4*height/5):int(height),0:int(width/7)]
	# show(small_area_bottom)
	# image = small_area_up
	shpMskValue = 140
	# if (debug_mode):show_image(small_area_bottom, window_name)

	bot,count,orientation=triangle(small_area_bottom,shpMskValue)
	if bot:
		if dictref is None and count==2:
			dictref==dicta
		elif dictref is None and count==1:
			dictref==dictb
		else:
			pass
		if orientation=='up':  
			rot_check==True
		elif orientation=='down':
			M = cv2.getRotationMatrix2D((c/2,r/2),180,1)
			crop = cv2.warpAffine(crop,M,(c,r))
			rot_check==True
		else:
			pass
	else:
		bot,count,orientation=triangle(small_area_up,shpMskValue)
		if bot:
			if dictref is None and count==2:
				dictref==dictb
			elif dictref is None and count==1:
				dictref==dicta
			else:
				pass
			if orientation=='up':
				rot_check==True
			elif orientation=='down':
				M = cv2.getRotationMatrix2D((c/2,r/2),180,1)
				crop = cv2.warpAffine(crop,M,(c,r))
				rot_check==True
			else:
				pass
	
	crop1=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
	ret,thresh=cv2.threshold(crop1,200, 255, 0)
	thr=threshold(crop)
	newtemp=threshold(temp)
	# print(thr.shape[0])
	# print(rows)
	# rows=int(rows*(float(1452)/1760))
	# print(rows)
	fx=float(thr.shape[1])/cols
	fy=float(thr.shape[0])/rows
	# print(temp.shape)
	newtemp=cv2.resize(newtemp,(int(fx*temp.shape[1]),int(fy*temp.shape[0])),cv2.INTER_AREA)
	# print(newtemp.shape)
	show_image(newtemp,window_name)
	crop,thresh,ims,logopos,maxval=tempmatch(thr,newtemp,thresh,crop,rot_check)
	logopos_act=dicta['temp']
	hrat=float(np.sqrt(((logopos[0][0]- logopos[1][0])**2)))/np.sqrt(((logopos_act[0][0]- logopos_act[1][0])**2))
	# vrat=hrat
	vrat=float(np.sqrt(((logopos[0][1]- logopos[1][1])**2)))/np.sqrt(((logopos_act[0][1]- logopos_act[1][1])**2))
	log=thresh[logopos[0][1]:logopos[1][1],logopos[0][0]:logopos[1][0]]
	
	hshift=int(((logopos[0][0]-logopos_act[0][0]*hrat)+(logopos[1][0]-logopos_act[1][0]*hrat))/2)
	vshift=int(((logopos[0][1]-logopos_act[0][1]*vrat)+(logopos[1][1]-logopos_act[1][1]*vrat))/2)
	mid_x_act=int(float(logopos_act[0][0]+logopos_act[1][0])/2)
	mid_y_act=int(float(logopos_act[0][1]+logopos_act[1][1])/2)
	mid_x=int(float(logopos[0][0]+logopos[1][0])/2)
	mid_y=int(float(logopos[0][1]+logopos[1][1])/2)
#print logopos,tuple([(int(x[0]*hrat),int(x[1]*vrat)) for x in logopos_act])    
#print hshift,vshift
	if dictref is None:
		formname=tuple([(int(x[0]*hrat),int(x[1]*vrat+vshift)) for x in dicta['form name']])
		newthr=ims[formname[0][1]:formname[1][1],formname[0][0]:formname[1][0]]
	#cv2.imwrite('/home/nishant/Desktop'+'/'+os.path.splitext(os.path.basename(filepath))[0]+str(np.random.randint(200))+'.jpg',newthr)
		pytesseract.pytesseract.tesseract_cmd='/usr/bin/tesseract'
	#Image.fromarray(newthr).show()
	  #  formname= image_to_string(Image.fromarray(thr[formname[0][1]:formname[1][1],formname[0][0]:formname[1][0]]))
		formname1= image_to_string(Image.fromarray(newthr),config='--psm=13')
		# printable='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

		# print(''.join(filter(lambda x: x in printable, formname1.lower())))

		
		for j in reversed(formname1[-7:]):
			# print(j)
			if j=='A':
				dictref=dicta
				break
			elif j=='B':
				dictref=dictb
				break
			# else:
			#     dictref=None
		if dictref is None:
			kernel = np.ones((3,3),np.uint8)
			newthr=thresh[formname[0][1]:formname[1][1],formname[0][0]:formname[1][0]]
			newthr=cv2.morphologyEx(newthr, cv2.MORPH_OPEN, kernel)
			formname= image_to_string(Image.fromarray(newthr),config='--psm=13')
			for j in reversed(formname):
					if j=='A':
						dictref=dicta
						break
					elif j=='B':
						dictref=dictb
						break
	dictnew={}
	# print(dictref)
	# print(mid_x_act,mid_x,mid_y_act,mid_y,hrat,vrat)
	vrat=1
	# hrat=1
	if dictref is not None:
		# print(dictref['pan'],vrat,vshift)
		for key in dictref:

			# print(key)
			dictnew[key]=tuple([(min(thr.shape[1],int((x[0]-mid_x_act+mid_x)*hrat)),int(min(thr.shape[0],(x[1]+mid_y-mid_y_act)*vrat))) if (x[0]<int(thr.shape[1]/2) and x[1]<int(2*thr.shape[0]/3)) else (min(thr.shape[1],int((x[0]-mid_x_act+mid_x)*hrat)),int(min(thr.shape[0],(x[1]+mid_y-mid_y_act)*vrat))) if (x[0]>int(thr.shape[1]/2) and x[1]<int(2*thr.shape[0]/3)) else (min(thr.shape[1],int((x[0]-mid_x_act+mid_x)*hrat)),int(min(thr.shape[0],(x[1]-mid_y_act+mid_y)*vrat))) if (x[0]<int(thr.shape[1]/2) and x[1]>int(2*thr.shape[0]/3)) else (min(thr.shape[1],int((x[0]-mid_x_act+mid_x)*hrat)),int(min(thr.shape[0],(x[1]-mid_y_act+mid_y)*vrat))) for x in dictref[key]])
			print(key,dictnew[key])
		# print(dictnew['cheque details'])	
		with open(outpath+os.path.splitext(os.path.basename(filepath))[0]+'.json','w') as f:
			json.dump(dictnew,f,separators=(',', ': '),indent=4)
		if highlightlist is not None:
			for words in highlightlist:
				cv2.rectangle(crop,dictnew[words][0],dictnew[words][1],(0,255,0),5)
			

		#print dictnew[words],tuple([(int(x[0]*hrat+hshift),int(x[1]*vrat+vshift)) for x in dictref[words]])
				
			show_image(crop,window_name)
	else:

		print("specify part")
		sys.exit()
		
			
			






