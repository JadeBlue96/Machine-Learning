
# coding: utf-8

# In[1]:

import cv2
import tensorflow as tf
import numpy as np
import os
import math
from PIL import Image,ImageFilter


# In[29]:

img=cv2.imread("samples/SAMPLE10.png")


# In[30]:

crop_img = img[233:388, 963:1097]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)


# In[130]:

im_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
_,ctrs,hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]


# In[131]:

i=0
for rect in rects:
    to_crop=crop_img[rect[1]:(rect[1] + rect[3]),rect[0]:(rect[0] + rect[2])]
    save_str='cropped_'+str(i)+'.png'
    save_dir=os.path.join('cropped_images/',save_str)
    cv2.imwrite(save_dir,to_crop)
    i=i+1
    #cv2.imwrite(save_dir,roi)
    # Draw the rectangles
    #cv2.rectangle(crop_img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1)


# In[69]:

cv2.imshow("Bounding boxes", crop_img)
cv2.waitKey(0)


# In[160]:

num_images=25
i=0
for i in range(num_images):
    newImage = Image.new('L', (128, 128), (255)) #creates white canvas of 28x28 pixels
    to_open_str='cropped_'+str(i)+'.png'
    open_dir=os.path.join('cropped_images/',to_open_str)
    im = Image.open(open_dir).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    
    if width > height: #check which dimension is bigger
        #Width is bigger. Width becomes 20 pixels.
        nheight = int(round((50.0/width*height),0)) #resize height according to ratio width
        if (nheight == 0): #rare case but minimum is 1 pixel
            nheight = 1  
        # resize and sharpen
        img = im.resize((50,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((128 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (40, wtop)) #paste resized image on white canvas
    else:
        #Height is bigger. Heigth becomes 20 pixels. 
        nwidth = int(round((50.0/height*width),0)) #resize width according to ratio height
        if (nwidth == 0): #rare case but minimum is 1 pixel
            nwidth = 1
         # resize and sharpen
        img = im.resize((nwidth,50), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((128 - nwidth)/2),0)) #caculate vertical pozition
        newImage.paste(img, (wleft, 40)) #paste resized image on white canvas
        
    to_save_str='test_'+str(i)+'.png'   
    save_dir=os.path.join('images_for_test/',to_save_str)
    newImage.save(save_dir)


# In[5]:

import glob
def image_preprocess(x_1,y_1,x_2,y_2,sample_path,save_cropped_path,save_test):
    image_ctr=0
    for filename in glob.glob(os.path.join(sample_path, '*.png')):
        img=cv2.imread(filename)
        crop_img = img[y_1:y_2, x_1:x_2]
        im_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
        ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
        _,ctrs,hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]
        for rect in rects:
            to_crop=crop_img[rect[1]:(rect[1] + rect[3]),rect[0]:(rect[0] + rect[2])]
            save_str='cropped_'+str(image_ctr)+'.png'
            save_dir=os.path.join(save_cropped_path,save_str)
            cv2.imwrite(save_dir,to_crop)
            image_ctr=image_ctr+1
            
        for i in range(image_ctr):
            newImage = Image.new('L', (128, 128), (255)) #creates white canvas of 28x28 pixels
            to_open_str='cropped_'+str(i)+'.png'
            open_dir=os.path.join(save_cropped_path,to_open_str)
            im = Image.open(open_dir).convert('L')
            width = float(im.size[0])
            height = float(im.size[1])
    
            if width > height: #check which dimension is bigger
                #Width is bigger. Width becomes 20 pixels.
                nheight = int(round((50.0/width*height),0)) #resize height according to ratio width
                if (nheight == 0): #rare case but minimum is 1 pixel
                    nheight = 1  
                # resize and sharpen
                img = im.resize((50,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
                wtop = int(round(((128 - nheight)/2),0)) #caculate horizontal pozition
                newImage.paste(img, (40, wtop)) #paste resized image on white canvas
            else:
                #Height is bigger. Heigth becomes 20 pixels. 
                nwidth = int(round((50.0/height*width),0)) #resize width according to ratio height
                if (nwidth == 0): #rare case but minimum is 1 pixel
                    nwidth = 1
                 # resize and sharpen
                img = im.resize((nwidth,50), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
                wleft = int(round(((128 - nwidth)/2),0)) #caculate vertical pozition
                newImage.paste(img, (wleft, 40)) #paste resized image on white canvas
        
            to_save_str='test_'+str(i)+'.png'   
            save_dir=os.path.join(save_test,to_save_str)
            newImage.save(save_dir)


# In[6]:

sample_path='samples/'
save_cropped_path='cropped_images/'
save_test='images_for_test/'
image_preprocess(x_1=963,y_1=233,x_2=1097,y_2=388,sample_path=sample_path,save_cropped_path=save_cropped_path,save_test=save_test)


# In[ ]:



