import os
import numpy as np

import chainer
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.datasets import TransformDataset
from chainer.datasets import cifar
import chainer.links as L
from chainer.training import extensions
from chainercv import transforms
import cv2 as cv
from skimage import transform as skimage_transform

def get_image_filepath_list(dir):
  file_path_list = []
  #クラスの抽出
  classes = sorted(os.listdir(dir))
  label_dic = {name:label for label, name in enumerate(classes)}

  for clas in classes:
      files = os.listdir(dir+clas)
      for file in files:
          file_path_list.append((dir+clas+"/"+file, label_dic[clas]))

  return file_path_list


USE_OPENCV = True


def cv_rotate(img, angle):
    if USE_OPENCV:
        img = img.transpose(1, 2, 0) / 255.
        center = (img.shape[0] // 2, img.shape[1] // 2)
        r = cv.getRotationMatrix2D(center, angle, 1.0)
        img = cv.warpAffine(img, r, img.shape[:2])
        img = img.transpose(2, 0, 1) * 255.
        img = img.astype(np.float32)
    else:
        # scikit-image's rotate function is almost 7x slower than OpenCV
        img = img.transpose(1, 2, 0) / 255.
        img = skimage_transform.rotate(img, angle, mode='edge')
        img = img.transpose(2, 0, 1) * 255.
        img = img.astype(np.float32)
    return img


def transform(inputs, random_angle=15., pca_sigma=255., train=True):
    img, label = inputs
    img = img.copy()

    #Random rotate
    # if train and random_angle != 0:
    #     angle = np.random.uniform(-random_angle, random_angle)
    #     img = cv_rotate(img, angle)

    if train:
        # random sized crop
        img = transforms.random_sized_crop(img)
        # Color augmentation
        img = transforms.pca_lighting(img, pca_sigma)
        # Random flip
        img = transforms.random_flip(img, x_random=True)

    # resize image
    img = transforms.resize(img, (128,128))
    # Standardization
    img /= 255.

    return img, label

if __name__ == "__main__":
    a = get_image_filepath_list("cats_and_dogs_small/train/")
    print(a)
