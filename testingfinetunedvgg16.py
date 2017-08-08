# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 12:30:15 2017

@author: Anoop
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 22:52:37 2017

@author: Anoop
"""
import os
from keras.preprocessing import image
from model import first_model
from vgg16finetuned_model import vgg16_finetuned
from keras.applications.vgg16 import preprocess_input 
import numpy as np
import cv2
import PIL
import matplotlib.pyplot as plt
#from nms import nms
from fastNms import non_max_suppression_fast
from nms import nms


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
    '''   #                         y = 1          x = 2
    #PIL image shape (channels, img_height, img_width)
    print image.shape
    for y in xrange(0, image.shape[1], step_size[1]):
        for x in xrange(0, image.shape[2], step_size[0]):
#            print (y + window_size[1], x + window_size[0])
#            print  image[:,y:y + window_size[1], x:x + window_size[0]].shape
            yield(x,y, image[:,y:y + window_size[1], x:x + window_size[0]])
            
            
            
#list to store the detections
detections = []
cd =[]
step_size = (25,25)
visualize = False

os.chdir('C:/Users/Anoop/Desktop/job_applications/IFM/')
#weights_path = 'C:/Users/Anoop/Desktop/job_applications/IFM/DeepLearning/weights/first_try.h5'
weights_path = 'DeepLearning/weights/vgg16_weights.h5'
#model = first_model()
#model.load_weights(weights_path)
model = vgg16_finetuned()
model.load_weights(weights_path)
img_width, img_height = 224, 224 # this is the model image size
input_shape = (3, img_width, img_height)
img_path = 'C:/Users/Anoop/Desktop/job_applications/IFM/testing/B.jpg'
img = image.load_img(img_path)  #same as img = PIL.Image.open(img_path)
img = image.img_to_array(img)   # converts to numpy float32 array
min_wdw_size = (75, 75)
downscale = 1.25
scale = 0

for (x,y, im_window) in sliding_window(img, min_wdw_size, step_size):
#        print "entered window loop"
#        print "im_window.shape", im_window.shape
#        print "min_wdw_size",min_wdw_size
        if im_window.shape[1] != min_wdw_size[1] and im_window.shape[2] != min_wdw_size[0]:
            print "size conflict, exiting the prediction process"
            break
        #2-tuple: (width, height).
        im_window = PIL.Image.fromarray(np.rollaxis(np.uint8(im_window), 0,3))
        im_window = im_window.resize((224,224), resample=PIL.Image.BILINEAR)
#        plt.imshow(np.asarray(im_window))
#        plt.show()
        im_window = image.img_to_array(im_window)
        im_window = np.expand_dims(im_window, axis=0)
        im_window = preprocess_input(im_window)
        pred = model.predict(im_window)
        print pred
        if pred > 0.9:
            print "Detection::Location ->({}, {})".format(x,y)           
#            detections.append((x,y, model.predict_proba(im_window), int(min_wdw_size[0]), int(min_wdw_size[1])))
            
            detections.append((x,y, pred,
                                       int(min_wdw_size[0]*(downscale**scale)),
                                       int(min_wdw_size[1]*(downscale**scale))))

#            detections.append((x,y, x+int(min_wdw_size[0])*(downscale**scale), y + int(min_wdw_size[1])*(downscale**scale)))

            cd.append(detections[-1])


clone =  preprocess_image(img_path)
##perform Non Maxima Suppression
#nmsdetections = non_max_suppression_fast(np.array(detections), 0.5)
#
## Display the results after performing NMS
#display_detections(clone ,nmsdetections)


 #perform Non Maxima Suppression
detections = nms(detections, threshold=0.5)

# Display the results after performing NMS

#    nmsdetections = non_max_suppression_fast(np.asarray(detections2,dtype = float), threshold)
#    display_detections(clone ,nmsdetections)

for (x_tl, y_tl, _, w, h) in detections:
    # Draw the detections
    cv2.rectangle(clone, (x_tl, y_tl), (x_tl+w,y_tl+h), (0, 0, 0), thickness=2)
       
cv2.imshow("Final Detections after applying NMS", clone)
cv2.imwrite("output.jpg", clone)
cv2.waitKey()
cv2.destroyAllWindows() 