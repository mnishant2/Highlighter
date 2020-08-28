## To run 
---
> Download the entire folder

Use cmd 

> **python** or **python3** *path/to/folder/highlighterfinal.py*  -f  *path/to/image/or/pdf/file.jpg*  -l  *list of words to find(can be left empty,e.g- date pin state will mark these three)*  -d  *(optional debug mode..if present,shows all intermediate steps)* 
 -o  *path/to/output/directory/ (where json with co-ordinates will be saved,if not explicitly defined,saves in the highlighter code directory)*

# sample
python highnew/highlighterfinal.py -f  /home/entrophy/Pictures/crm.jpeg  -l date pin state -d


To check the field names look into the crfform xml files included in the folder

If the field names or file names have spaces,eg 'customer name',write on terminal like customer\ name

Packages needed

1. Tesseract-ocr

2. Pytesseract

3. Opencv

4. Pdf2image

5. PIL

6. Skimage


Once it runs it will prompt you if you do not have any package.

A json file will be created in the specified output folder with all coordinates of various fields after the execution.The output file will have same name plus index as input file
### If there is an error it will most probably be because of wrong ocr.
### Ideal for 150 dpi scans
Also you might have to tinker the tesseract file address in code to the appropriate address in your computer.**At present it is /usr/bin/tesseract in line 249 of code.**

**To locate tesseract use *'which tesseract' or 'locate tesseract'* commands on terminal**

## The co-ordinates returned will be with respect to the rotated image named crop.Hence return it for further augmentation into the application.