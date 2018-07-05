#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:16:45 2018

@author: wu
"""

import os
import tensorflow as tf
from PIL import Image
import numpy as np
import math

path = '/home/wu/TF_Project/flower/flower_photos/'
ratio = 0.2

def get_files(file_dir, ratio):
    class_train = []
    label_train = []
    k = 0
    for train_class in os.listdir(file_dir):
        for sub_train in os.listdir(file_dir+train_class):
            for image in os.listdir(file_dir+train_class+'/'+sub_train) :
                class_train.append(file_dir+train_class+'/'+sub_train+'/'+image)
                label_train.append(k)
        k+=1
    temp = np.array([class_train,label_train])
    temp = temp.transpose()
    #shuffle the samples
    np.random.shuffle(temp)
    #after transpose, images is in dimension 0 and label in dimension 1
    all_image_list = list(temp[:,0])
    all_label_list = list(temp[:,1])
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample*ratio)) #测试样本数
    n_train = n_sample - n_val # 训练样本数
    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]
    return tra_images,tra_labels,val_images,val_labels
    
#制作二进制数据
def create_record():
    writer_train = tf.python_io.TFRecordWriter('/home/wu/TF_Project/flower/flower_TFrecord/train.tfrecords')
    writer_val = tf.python_io.TFRecordWriter('/home/wu/TF_Project/flower/flower_TFrecord/val.tfrecords')
    tra_images,tra_labels,val_images,val_labels = get_files(path, ratio)
    
    for index, name in enumerate(tra_images):
        img = Image.open(name)
        img = img.resize((100, 100))
        img_raw = img.tobytes()
        example_train = tf.train.Example(
           features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[tra_labels[index]])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                 }))
        writer_train.write(example_train.SerializeToString())
    for index, name in enumerate(val_images):
        img = Image.open(name)
        img = img.resize((100, 100))
        img_raw = img.tobytes()
        example_val = tf.train.Example(
           features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[val_labels[index]])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                 }))
        writer_val.write(example_val.SerializeToString())

    writer_train.close()
    writer_val.close()

data = create_record()