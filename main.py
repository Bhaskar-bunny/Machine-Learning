from fpt import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

# ap= argparse.ArgumentParser()
# ap.add_argument("-i","--image",required=True,
#                 help="Path to the image to be scanned")
# args=vars(ap.parse_args())

image=cv2.imread('Sample.jpg')
ratio=image.shape[0]/500.0
orig = image.copy()
image=imutils.resize(image,height=500)

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray=cv2.GaussianBlur(gray,(5,5),0)
edged=cv2.Canny(gray,75,200)

print("step 1: Edge detection")
# cv2.imshow("Image",image)
# cv2.imshow("Edged",edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cnts=cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)
cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:5]

for c in cnts:
    peri=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.02*peri,True)

    if len(approx)==4:
        screenCnt=approx
        break

print("Step 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1,(0,255,0),2)
# cv2.imshow("Outline",image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


warped=four_point_transform(orig, screenCnt.reshape(4,2)*ratio)

warped=cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
T=threshold_local(warped,11,offset=10,method="gaussian")
warped=(warped > T).astype("uint8")*255

print("Step 3: Apply perspective transform")

imS=cv2.resize(warped,(650,650))
# cv2.imshow("output",imS)
# cv2.imwrite('Output Image.png',imS)
# cv2.waitKey(0)

from PIL import Image
import PIL.Image
from pytesseract import image_to_string
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSDATA_PREFIX = 'C:/Program Files /Tesseract-OCR'
output = pytesseract.image_to_string(PIL.Image.open('Output Image.png').convert("RGB"), lang='eng')
# print(output)

f = open('r.txt','w')
f.write(output)
f.close()

import nltk

from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import ne_chunk

raw= open('r.txt').read()

Ne_tokens=word_tokenize(raw,"english",True)

Ne_tags= pos_tag(Ne_tokens)
# print(Ne_tags)

Ne_ner= ne_chunk(Ne_tags)
# print(Ne_ner)
Ne=[]

for x,y in Ne_tags:
     if y == 'NNP':
         Ne.append((x,y))

F=open('AAA.txt','w')

from itertools import groupby
for tag, chunk in groupby(Ne, lambda x:x[1]):
    if tag != "O":
        F.write(" ".join(w for w, t in chunk)+"\n")






import re

#regular expression to find emails
emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", output)
#regular expression to find phone numbers
numbers = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', output)

print(numbers)
print(emails)

for email in emails:
	print('EMAIL :-> ' + email)
	F = open('AAA.txt','a+')
	F.write('EMAIL :-> ' + email)

for number in numbers:
	print('Phone No. :-> ' + number)
	F = open('AAA.txt', 'a+')
	F.write('\n Phone No. :-> ' + number)