
# coding: utf-8

# In[1]:




# In[3]:

import os
import glob
import numpy as np
import cv2
from sklearn.utils import shuffle
from cache import cache
from PIL import Image, ImageFilter

def load_train(train_path, image_size, classes):
    images = []
    labels = []
    ids = []
    cls = []

    print('Reading training images')
    for fld in classes:   # assuming data directory has a separate folder for each class, and that each folder is named after the class
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(train_path, fld, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            ids.append(flbase)
            cls.append(fld)
    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)

    return images, labels, ids, cls


def load_test(test_path, image_size,classes):
  X_test = []
  X_test_id = []
  X_test_labels=[]
  X_test_cls=[]
  print("Reading test images")
  for fld in classes:   # assuming data directory has a separate folder for each class, and that each folder is named after the class
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(test_path, fld, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
            X_test.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            X_test_labels.append(label)
            flbase = os.path.basename(fl)
            X_test_id.append(flbase)
            X_test_cls.append(fld)
  X_test = np.array(X_test)
  X_test_labels = np.array(X_test_labels)
  X_test_id = np.array(X_test_id)
  X_test_cls = np.array(X_test_cls)

  return X_test, X_test_labels,X_test_id,X_test_cls

import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def load_custom_test(custom_path,image_size,classes):
  X_test = []
  X_test_id = []
  X_test_labels=[]
  X_test_cls=[]
  print("Reading test images")
  index = 0
  for fl in sorted(glob.glob(os.path.join(custom_path, '*.png')),key=numericalSort):
          image = cv2.imread(fl)
          image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
          X_test.append(image)
          label = np.zeros(len(classes))
          label[index] = 1.0
          X_test_labels.append(label)
          flbase = os.path.basename(fl)
          X_test_id.append(flbase)
          X_test_cls.append('0')
  X_test = np.array(X_test)
  X_test_labels = np.array(X_test_labels)
  X_test_id = np.array(X_test_id)
  X_test_cls = np.array(X_test_cls)

  return X_test, X_test_labels,X_test_id,X_test_cls



class DataSet(object):

  def __init__(self, images, labels, ids, cls):

   
    self._num_examples = images.shape[0]

    #images = images.astype(np.float32)
    #images = np.multiply(images, 1.0 / 255.0)

    self._images = images
    self._labels = labels
    self._ids = ids
    self._cls = cls
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def ids(self):
    return self._ids

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):

    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1

      # # Shuffle the data 
      # perm = np.arange(self._num_examples)
      # np.random.shuffle(perm)
      # self._images = self._images[perm]
      # self._labels = self._labels[perm]
      # Start next epoch

      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._ids[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes):
  class DataSets(object):
    pass
  data_sets = DataSets()
  images, labels, ids, cls = load_train(train_path, image_size,classes)
  images, labels, ids, cls = shuffle(images, labels, ids, cls)  # shuffle the data

  train_images = images
  train_labels = labels
  train_ids = ids
  train_cls = cls

  data_sets.train = DataSet(train_images, train_labels, train_ids, train_cls)
  
  return data_sets


def read_test_set(test_path, image_size,classes):
  class DataSets(object):
    pass
  data_sets = DataSets()  
  images,labels, ids,cls  = load_test(test_path, image_size,classes)
  data_sets.test = DataSet(images, labels, ids, cls)
  return data_sets

def read_custom_test(custom_path,image_size,classes):
    class DataSets(object):
        pass
    data_sets=DataSets()
    images,labels,ids,cls=load_custom_test(custom_path,image_size,classes)
    data_sets.test=DataSet(images,labels,ids,cls)
    return data_sets


# In[ ]:


# In[ ]:



