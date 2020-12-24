import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


class EnFCM():
    def __init__(self, image, image_bit, n_clusters, m, neighbour_effect, epsilon, max_iter, kernel_size):
            self.image = image
            self.image_bit = image_bit
            self.n_clusters = n_clusters
            self.m = m
            self.neighbour_effect = neighbour_effect
            self.epsilon = epsilon
            self.max_iter = max_iter
            self.kernel_size = kernel_size

            #Attributes according to image.
            self.shape = image.shape # image shape
            self.X = image.flatten().astype('float') # flatted image shape: (number of pixels,1)
            self.numPixels = image.size

    def initial_U(self):
        '''Initialization Membership matrix '''
        U=np.zeros((self.num_gray, self.n_clusters))
        idx = np.arange(self.num_gray)
        for ii in range(self.n_clusters):
            idxii = idx%self.n_clusters==ii
            U[idxii,ii] = 1
        return U

    def update_U(self):
        '''Compute weights'''
        idx = np.arange(self.num_gray)
        c_mesh,idx_mesh = np.meshgrid(self.C,idx)
        power = -2./(self.m-1)
        numerator = abs(idx_mesh-c_mesh)**power
        denominator = np.sum(abs(idx_mesh-c_mesh)**power,axis=1)
        return numerator/denominator[:,None]

    def update_C(self):
        '''Compute centroid of clusters'''
        idx = np.arange(self.num_gray)
        idx_reshape = idx.reshape(len(idx),1)
        numerator = np.sum(self.histogram*idx_reshape*pow(self.U,self.m),axis=0)
        denominator = np.sum(self.histogram*pow(self.U,self.m),axis=0)
        return numerator/denominator

    ''' Until now we did nothing new according to classical FCM '''
    ''' We begin the unhanced version of FCM '''
    def get_filtered_image(self):
        #Create padding image
        print("Getting filtered image..")
        mask = np.ones((self.kernel_size,self.kernel_size))
        mask[int(self.kernel_size/2),int(self.kernel_size/2)]=0
        a = self.neighbour_effect
        mean_image = get_mean_image_in_window(self.image, mask)
        filtered_image = (self.image+a*mean_image)/(1+a) #Somme linéaire du poid de l'image
        dtype = self.image.dtype
        self.filtered_image = filtered_image.reshape(self.shape).astype(dtype)

    def calculate_histogram(self):
        hist_max_value = (1 << self.image_bit)
        hist = cv2.calcHist([self.filtered_image],[0],None,[hist_max_value],[0,hist_max_value])
        self.num_gray = len(hist)
        self.histogram = hist

    def form_clusters(self):
        ''' Form clusters and train model '''
        self.get_filtered_image()
        self.calculate_histogram()
        '''Iterative training'''
        d = 100
        self.U = self.initial_U()
        if self.max_iter != -1:
            i = 0
            while True:
                self.C = self.update_C()
                old_u = np.copy(self.U)
                self.U = self.update_U()
                d = np.sum(abs(self.U - old_u))
                print("Iteration %d : cost = %f" %(i, d))
                if d < self.epsilon or i > self.max_iter:
                    break
                i+=1
        else:
            i = 0
            while d > self.epsilon:
                self.C = self.update_C()
                old_u = np.copy(self.U)
                self.U = self.update_U()
                d = np.sum(abs(self.U - old_u))
                print("Iteration %d : cost = %f" %(i, d))

                if d < self.epsilon or i > self.max_iter:
                    break
                i+=1
        self.segmentImage()
    def deFuzzify(self):
        return np.argmax(self.U, axis = 1)

    def segmentImage(self):
        '''Segment image based on max weights'''
        result = self.deFuzzify()
        self.result = np.array(self.image, copy=True)
        for i in range(len(result)):
            self.result[self.result==i]=result[i]
        self.result = self.result.reshape(self.shape).astype('int')
        return self.result

def get_mean_image_in_window(image, kernel):
    '''Get image consisting of mean values ​​of neighboring pixels in a window '''
    neighbor_sum = convolve2d(
        image, kernel, mode='same',
        boundary='fill', fillvalue=0)
    num_neighbor = convolve2d(
        np.ones(image.shape), kernel, mode='same',
        boundary='fill', fillvalue=0)
    return neighbor_sum / num_neighbor

#### Main ####
#--------------Lord image file--------------
#img= cv2.imread('./img/peppers.png', cv2.IMREAD_GRAYSCALE) # cf. 8bit image-> 0~255
img= cv2.imread('./img/gwpe.png', cv2.IMREAD_GRAYSCALE) # cf. 8bit image-> 0~255

#--------------Clustering--------------
cluster = EnFCM(img, image_bit=8, n_clusters=4, m=2, neighbour_effect=2, epsilon=0.05, max_iter=100, kernel_size=4)
cluster.form_clusters()
result=cluster.result

#--------------- Plot ------------------------
fig=plt.figure(figsize=(12,8),dpi=100)

ax1=fig.add_subplot(1,2,1)
ax1.imshow(img,cmap='gray')
ax1.set_title('image')

ax2=fig.add_subplot(1,2,2)
ax2.imshow(result)
ax2.set_title('segmentation')

plt.show()
