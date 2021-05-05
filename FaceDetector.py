import cv2
import glob
import numpy as np
import json
import os
import sys

#cascade file
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
images = []
image_name=[]
#################----PATH---############################
path=sys.argv[1]
# print(path)

#Reading of images
for img in glob.glob(path+"/*.jpg"):
  n= cv2.imread(img, cv2.IMREAD_GRAYSCALE)
  #n=n-10 #changing intensity for better matches
  image_name.append(os.path.basename(img))
  images.append(n)
# print(len(image_name))
# imgpath = [f'./Validation folder/images/*.jpg' for n in range(1,N+1)]
#     imgs = []
#     for ipath in imgpath:
#         img = cv2.imread(ipath)
#         imgs.append(img)

#detecting faces using faceCascade file
facefinal=[]
bbox=[]
imgname=[]
for m in range(len(images)):

  #face =faceCascade.detectMultiScale(gray_image[m], 1.3, 4)
  facee = faceCascade.detectMultiScale(images[m], 1.2, 5)
  face = [x + 2 for x in facee] #increasing bbox by 2 units for better detection

  for ll in np.array(face).tolist():
    imgname.append(image_name[m])
    bbox.append(ll)

# dictionary=dict(zip(imgname,bbox))
# print(dictionary)

#making list to dumpt to json
jsonList=[]
for i in range(0,len(bbox)):
    jsonList.append({"iname" : imgname[i], "bbox" : bbox[i]})


#json dump
output_json = "results.json"
#dump json_list to result.json
with open(output_json, 'w') as f:
    json.dump(jsonList, f)

# print(face)
# for (x,y,h,w) in face:
#   cv2.rectangle(image, (x,y) , (x+w, y+h), (255,255,0),2)
#   cv2.imshow("img",image)
#   cv2.waitKey()
