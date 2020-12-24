#!/usr/bin/env python
# ## Implementation de FCM Fuzzy C-Means de Scratch
# ### Importation des librairies
import os
import random
import numpy as np
import matplotlib.image as img
import copy
from PIL import Image

# #### Définition des paramètres pour FCM
K = 2
m = 2
epsilon = 10

# #### Lecture de la donnée
data = img.imread('./image/rock1.jpg')
img_shape = data.shape
data = data.flatten()
data = [[x] for x in data]
N = len(data)

def euclid_dist(x,y):
    return sum([(i-j) for i,j in zip(x,y)])

def updateMembershipValue(U,C):
    new_U = copy.deepcopy(U)
    dominateur=0
    p = 2/(m-1)
    for i in range(N):
        for j in range(K):
            dominateur =0
            for k in range(K):
                dominateur += pow(euclid_dist(data[i],C[j])/euclid_dist(data[i],C[k]),p)
            new_U[i][j] = float(1/dominateur)
    return new_U

def calculateClusterCenter(U):
    C = []
    for j in range(K):
        temp,temp1,temp2=0,[],[]
        for i in range(N):
            temp += U[i][j]
            temp2 = [U[i][j]*m for val in data[i]]
            temp1.append(temp2)
        numerator = list(map(sum,zip(*temp1)))
        C.append([x/temp for x in numerator])
    return C

def stop_Constrain(U, new_U):
    z = np.subtract(U,new_U)
    z = np.absolute(z)
    max_ = np.max(z)
    return max_ < epsilon

def getClusters(U):
    cluster_labels = list()
    for i in range(N):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(U[i]))
        cluster_labels.append(idx)
    return cluster_labels

def initializeMembershipMatrix():
    U = list()
    for i in range(N):
        random_num_list = [random.random() for i in range(K)]
        s= sum(random_num_list)
        temp_list = [x/s for x in random_num_list]
        U.append(temp_list)
    return U

def fuzzy_Classification():
    U = initializeMembershipMatrix()
    while (1):
        C = calculateClusterCenter(U)
        new_U = updateMembershipValue(U, C)
        if(stop_Constrain(U,new_U)):
            clusters = 255*np.array(getClusters(new_U))
            clusters = clusters.reshape(img_shape)
            img = Image.fromarray(clusters.astype(np.uint8))
            img.show()
            break

fuzzy_Classification()
