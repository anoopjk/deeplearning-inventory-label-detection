# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 22:52:37 2017

@author: Anoop
"""
import os
from model import first_model
from vgg16finetuned_model import vgg16_finetuned
from keras.applications.vgg16 import preprocess_input 
import numpy as np
import cv2
import argparse as ap
#from nms import nms
from fastNms import non_max_suppression_fast



from skimage.transform import rescale, resize
from skimage.transform import pyramid_gaussian

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    img = cv2.resize(img, (640, 480), interpolation = cv2.INTER_AREA)
#    img = cv2.convertScaleAbs(img)
    
    return img
    
    
def display_detections(clone ,nmsdetections):
 
    if not len(nmsdetections) == 0:
        for  i in  range(nmsdetections.shape[0]):
            # Draw the detections
            x_tl = nmsdetections[i, 0]
            y_tl = nmsdetections[i, 1]
            x_br = nmsdetections[i, 2]
            y_br = nmsdetections[i, 3]
        
            cv2.rectangle(clone, (x_tl, y_tl), (x_br,y_br), (0, 0, 0), thickness=2)   
        cv2.imshow("Final Detections after applying NMS", clone)
        cv2.imwrite('nms_output.png', clone)
        cv2.waitKey()
        cv2.destroyAllWindows() 


def sliding_window(image, window_size, step_size):
    '''
    This function return a patch of the input image of size equal to window_size. 
    the first image returned top-left coordinates(0,0) and are incremented in both
    x and y directions by the 'step_size' supplied.
    the input parameters are- 
    *'image' - Input image
    *'window_size' -Size of the Sliding Window
    *'step_size' - incremented size of the window
    
    THe function returns a tuple -
    (x,y, im_window)
    where 
    *x is top-left x coordinate
    *y is top-left y coordinate
    *im_window is the slding window image
    '''
    for y in xrange(0, image.shape[0], step_size[1]):
        for x in xrange(0, image.shape[1], step_size[0]):
            yield(x,y, image[y:y + window_size[1], x:x + window_size[0]])
            
            
            
#list to store the detections
detections = []
cd =[]
step_size = (5,5)
visualize = True

os.chdir('C:/Users/Anoop/Desktop/job_applications/IFM/')
#weights_path = 'C:/Users/Anoop/Desktop/job_applications/IFM/DeepLearning/weights/first_try.h5'
weights_path = 'DeepLearning/weights/vgg16_weights.h5'
#model = first_model()
#model.load_weights(weights_path)
model = vgg16_finetuned()
model.load_weights(weights_path)
img_width, img_height = 224, 224 # this is the model image size
input_shape = (3, img_width, img_height)
img_path = 'C:/Users/Anoop/Desktop/job_applications/IFM/testing/Bhalfscale.jpg'
#img = image.load_img(img_path, target_size=(img_width, img_height))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
img = preprocess_image(img_path)



min_wdw_size = (10, 100)
downscale = 1.25
scale = 0
#print min_wdw_size
for i in range(0,3):
    im_scaled = cv2.resize(img, (int(img.shape[1]/(1+ 0.25*scale)), int(img.shape[0]/(1 +  0.25*scale))), interpolation = cv2.INTER_AREA)
#    print im_scaled
    for (x,y, im_window) in sliding_window(im_scaled, min_wdw_size, step_size):
    #        print "entered window loop"
    #                print "im_window.shape", im_window.shape
    #                print "min_wdw_size",min_wdw_size
            if im_window.shape[0] != min_wdw_size[1] or im_window.shape[1] != min_wdw_size[0]:
                print "size conflict, exiting the prediction process"
                break
            
            im_window = cv2.resize(im_window, (img_width, img_height), interpolation = cv2.INTER_LINEAR)
            im_window = im_window.astype('float64')
            im_window = np.transpose(im_window, (2, 1, 0))
            im_window = np.expand_dims(im_window, axis=0)
            im_window = preprocess_input(im_window)
            pred = model.predict(im_window)
            print pred
            if pred > 0.9:
                print "Detection::Location ->({}, {})".format(x,y)           
    #            detections.append((x,y, model.predict_proba(im_window), int(min_wdw_size[0]), int(min_wdw_size[1])))
                                   
                detections.append((x,y, x+int(min_wdw_size[0])*(downscale**scale), y + int(min_wdw_size[1])*(downscale**scale)))
                                   
                cd.append(detections[-1])
                
            #if visualize is set to true, display the working
            #of the sliding window
            if visualize == True:
                clone = img.copy()
                for x1, y1, x2, y2 in cd:
                    #Draw the detections at this scale
                    cv2.rectangle(clone, (x1,y1), (x2, y2),(0,0,0), thickness=2)
                    cv2.rectangle(clone, (x,y), (x+ im_window.shape[1], y+ im_window.shape[0]),
                                  (255, 255, 255), thickness= 2)
                    cv2.imshow("sliding window in progress", clone)
                    cv2.waitKey(30)
    scale +=1
 

clone = img.copy()
#perform Non Maxima Suppression
nmsdetections = non_max_suppression_fast(detections, 0.5)

# Display the results after performing NMS
display_detections(img ,nmsdetections)


#############################################################################
from keras.preprocessing import image
import os
from vgg16finetuned_model import vgg16_finetuned
from keras.applications.vgg16 import preprocess_input 
import numpy as np

weights_path = 'C:/Users/Anoop/Desktop/job_applications/IFM/DeepLearning/weights/vgg16_weights.h5'
model = vgg16_finetuned()
model.load_weights(weights_path)
img_width, img_height = 224, 224 # this is the model image size
img_path = 'C:/Users/Anoop/Desktop/job_applications/IFM/testing/tile1.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_AREA)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img= img.astype('float64')

img = np.transpose(img, (2, 1, 0))
img= np.expand_dims(img, axis=0)
x = preprocess_input(img)
#img = image.load_img(img_path, target_size=(img_width, img_height))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)
pred = model.predict(x)
print pred