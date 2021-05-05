#!pip3 install face_recognition
#from google.colab import drive
#drive.mount('/content/drive')


import face_recognition
import cv2
import glob
import numpy as np
import json
import os
import face_recognition
from sklearn.metrics import pairwise_distances_argmin
#from google.colab.patches import cv2_imshow
import sys

#PATH of cascade file
faceCascade = cv2.CascadeClassifier('Model_Files/haarcascade_frontalface_default.xml')
images = []
image_name=[]
list_test =[]

#PATH OF IMAGE FOLDER || generic path ||
path_path=sys.argv[1]
# print(path)
# path_path="faceCluster_5"
#for img in glob.glob(path+"/*.jpg"):
print(path_path)

for img in glob.glob(path_path+"/*.jpg"):
  # print(img)
  n= cv2.imread(img)
  #cv2_imshow(n)
  #cv2.waitKey(5)
  # n=n+10
  image_name.append(os.path.basename(img))
  list_test.append(img)
  images.append(n)
count=0
#print(len(images))
bboxx=[]
crop_face=[]
for m in range(len(images)):
  #face =faceCascade.detectMultiScale(gray_image[m], 1.1, 4)
  face = faceCascade.detectMultiScale(images[m], 1.3, 5)
  for (x,y,w,h) in face:
    #cv2.rectangle(images[k],(x, y), (x+w, y+h), (0, 255, 255), 2)
    # #cropping
    #print(count,"+++",x,y,w,h)
    count=count+1
    bboxx.append(images[m][y:y + h, x:x + w])
  # print(bboxx)
# count
dict_temp = {}
if os.path.exists("Cropimg"):
    for f in glob.glob("Cropimg/*.jpg"):
        os.remove(f)
else:
    os.mkdir("Cropimg")
for i in range(len(bboxx)):
  # 3
  #cv2_imshow(bboxx[i])
  #cv2.waitKey(1)

  #temp folder = Cropimg
  filename = "Cropimg/" + str(i + 1) + '.jpg'
  dict_temp[filename] = list_test[i]
  cv2.imwrite(filename, bboxx[i])

  #print(filename)
#os.mkdir("Cropimg")

face_enc=[]
for img in glob.glob("Cropimg/*.jpg"):
  #print(img)
  image = face_recognition.load_image_file(img)
  image=image-10 #changing intensity of images for better matches
  face_enc.append(face_recognition.face_encodings(image))

#checking number of cluster from folder name
n=path_path
#n="faceCluster_5"
kk=int(n[-1])
#print(kk)

face_array=np.asarray(face_enc)
face_new_array=face_array.reshape(count,128)

#clustering
center_c = []

X=face_new_array
number_of_clusters=kk
rseed=kk
#finding the best centroid and looping till convergence of distances to best clusters
center_c.append(X[np.random.randint(X.shape[0]), :])
for mmm in range(kk - 1):
  distancess = []
  for i in range(X.shape[0]):
    data_points = X[i, :]
    dist = sys.maxsize
    for j in range(len(center_c)):
        current_distance = np.sum((data_points - center_c[j])**2)
        dist = min(dist, current_distance)
    distancess.append(dist)

  distancess = np.array(distancess)
  center_c.append(X[np.argmax(dist), :])
random_clusters = np.random.RandomState(rseed)
cluster_perm = random_clusters.permutation(X.shape[0])[:number_of_clusters]
center_c = X[cluster_perm]
while True:
    label_faces = pairwise_distances_argmin(X, center_c)
    #label_faces1 = pairwise_distances_argmin(X)
    new_center_c = np.array([X[label_faces == cluster_perm].mean(0) for cluster_perm in range(number_of_clusters)])
    #new_label_faces = np.array([X[label_faces == cluster_perm].mean(0) for cluster_perm in range(number_of_label_faces)])
    #convergence check
    if np.all(center_c == new_center_c):
        break
    center_c = new_center_c


#label_faces is the final labels
list_unique = np.unique(label_faces)
dict_indicies ={}
for k in list_unique:
  list_temp =[]
  for i in range(label_faces.size):
      if label_faces[i] == k:
        list_temp.append(i)
  dict_indicies[k] = list_temp

dict_indicies

img111 = []
json_list = []
list_paths = glob.glob("Cropimg/*.jpg")
# list_path_1=glob.glob("/content/drive/MyDrive/50373843/faceCluster_5/*.jpg")
#  print(img)
for key, value in dict_indicies.items():
  list_arrays = []
  list_arrays1 = []
  list_image_name = []
  print("Cluster :" + str(key + 1))
  for m in dict_indicies[key]:
    p = cv2.imread(list_paths[m])
    p1 = cv2.imread(dict_temp[list_paths[m]])
    list_image_name.append(dict_temp[list_paths[m]].split('/')[-1])
    # print(list_image_name)
    p = cv2.resize(p, (180, 180))
    p1 = cv2.resize(p1, (180, 180))
    list_arrays.append(p)
    list_arrays1.append(p1)
  temp = tuple(list_arrays)
  temp1 = tuple(list_arrays1)
  # cv2_imshow(np.hstack(temp))
  # print(list_path_1)
  #cv2_imshow(np.hstack(temp1))
  cv2.imshow('img',np.hstack(temp1))
  cv2.waitKey(0)

  json_list.append({"cluster_no": int(key), "elements": [i for i in list_image_name]})

# output_json = "results.json"
# #dump json_list to result.json
# with open(output_json, 'w') as f:
#     json.dump(jsonList, f)

output_json = "clusters.json"
#dump json_list to result.json
with open(output_json, 'w') as f:
    json.dump(json_list, f)