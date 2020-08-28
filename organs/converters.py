'''
This script takes in a converts input numpy image list into a pdf

1. First it write the files into a tmp folder
2. Second reads the image file in that folder
3. Form a list of image filepaths
4. Converts into a pdf
5. Writes the pdf file to the folder
5. Deletes the images in the folder

'''
######### have to include unique cis number in the file name

from fpdf import FPDF
import cv2
import numpy as np
import os
import sys
import base64
from PIL import Image
from io import StringIO


def convertToPdf(imagelist, cif_number, filename):
    print ("###############################################")
    print ("pdf conversion initiated....")
    pdf = FPDF()
    for i , image in enumerate(imagelist):
        name = "tmp/"+str(cif_number)+"/forms/"+filename+str(i)+".jpg"
        cv2.imwrite(name,image)

    # imagelist is the list with all image filenames
    fileList = []
    for file in os.listdir("tmp/"+str(cif_number)+"/forms/"):
        if (file.endswith(".jpg") or file.endswith(".jpeg")):
            full_file_name = os.path.join('tmp/'+str(cif_number)+"/forms/", file)
            #print (full_file_name)
            fileList.append(full_file_name)

    xMargin=10
    yMargin=10
    w=200
    h=200
    fileList.sort()
    for image in fileList:
        pdf.add_page()
        pdf.image(image,xMargin,yMargin,w,h)
    pdf.output("tmp/"+str(cif_number)+"/forms/"+str(cif_number)+"_"+filename+".pdf", "F")
    fileList = []
    for file in os.listdir("tmp/"+str(cif_number)+"/forms"):
        if file.endswith(".jpg") or file.endswith(".jpeg"):
            full_file_name = os.path.join('tmp/'+str(cif_number)+"/forms", file)
            os.remove(full_file_name)

    print ("pdf converted")



def convertBaseb64ToCv2(base64_string):
    img = base64.b64decode(base64_string); 
    npimg = np.fromstring(img, dtype=np.uint8); 
    cv2Image = cv2.imdecode(npimg, 1)
    return cv2Image
    



if __name__ == "__main__":
    #try
    # imagesSet = ["/home/yuyudhan/abs/signzySampleImages/aadhaar-basic/9ca16a7f30c981fe892ab4ab9dc093734390bfb9803379f4d0c470d8c6443ab2.jpeg",
    # "/home/yuyudhan/abs/signzySampleImages/aadhaar-basic/16d1bb8bcf816fdf913583f74f4262eb5190c3e253d9e59f4df6c164b171049e.jpg"]
    # numpyImagesSet = []
    # for i in range(0, len(imagesSet)):
    # 	try:
    # 		numpyImagesSet.append(cv2.imread(imagesSet[i]))
    # 		pass
    # 	except Exception as e:
    # 		raise e
    # 	pass
    # convertToPdf(numpyImagesSet,1)


    #######
    #for convertBaseb64ToCv2 function
    with open('/home/signzy-engine/Downloads/IMG_20171115_232354.jpg', "rb") as zip_file:
        data = zip_file.read()
        base64_bytes = base64.b64encode(data)
        base64_string = base64_bytes.decode('utf-8')        
    
    base64_string = "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88+JKCMjIzAwUNi2Hzx4QETbt2+/dOnStWvXOnXqREQbNmxgOz6J5GGeXrx48QcffJCVlRUeHs4S0qVLl2JjY1EraomPj7916xYRTZkyhYgmT57MGlEiIiKMZxNWa0ZGRmJi4tGjR8+dO8dq19WrV3fu3PlIszVGSEgIizgjR468evVqQUHB/PnziSgrK+sf//gHW+llZWVszpSUlMzMTHt7eyLatm3bvXv3iGj37t1ff/01W9rs2bOzs7ODgoJMftamTZtYxPHw8EhLSyssLDx16lTPnj3ZQqKioh6eq/1Ru4KCgnx9fU+dOrVt2zZWu65cuSJsC99++61Op7O2tj537lxOTk5+fj4bjnru3LlDhw6ZWwWYOHEie7Fx40ZPT88VK1YcPny4qKio7pxff/01OyQHBgampaWlp6ezDXPPnj1xcXEmS7vxq6CRZRscHMy2dB8fn8uXLxcUFCxbtkxokxNuE/d0V8Hp06e3/tnhw4cftTxNLrkFty9AymkLDAbDxo0bRSJRVVUVO0LUwvM8SxsymSwgIIBNXLRoUbt27VjQEVKOkGNOnDhhMBj69evHzp/i4+OJ6M6dO7m5ucYz13L//n3WvCTsPWfMmFFYWKjVav38/NiU7777Tji8WVtbSySS5cuXz5o1a+bMmcIecM6cOfHx8fHx8W+//TY7DfXy8mK/NCcnx/gTX3755ZCQkL59+77zzjuzZs1iEy9evIhaUQvrnLKwsBg7diwRDRgwoFu3bmSq04qZPn16r1692Gmlv78/m8gCcRNma4AwHGT16tWsPixevPjtt99+8803WX147733oqOjo6Ojp06dSkQqlWrQoEGsMly+fPmRCkGoe+vXr2fhxsXFZfXq1WxieHh4rfnd3d1Xr17t4uIybdq0t956i00UPpQ1XVRXV9++fVsojZycnNu3bwszmw9vb+/169d36NCBiHJyckJCQt566y1nZ2cPD4/AwEChn5fn+T179rB9xSeffMImzps3j+0rfvjhh5b6Pg2XbUxMjHB6ZmtrK5PJ5s6d+9JLL9VayNNdBTExMZ//2a5du+rue5tTns3fvqDFocfq6XB3d3///fe3b99+8ODB2NjY119/3fi/RUVFbC8gEomMr3TgOI6IWBtP7969HRwcbt68efbs2Y8++uiXX35hJ1JspA47xzp79ix7Y63lC8aPH5+S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"
    img = base64.b64decode(base64_string); 
    npimg = np.fromstring(img, dtype=np.uint8); 
    source = cv2.imdecode(npimg, 1)

    #cvimg = convertBaseb64ToCv2(base64_string)
    cv2.imshow("image",source)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
