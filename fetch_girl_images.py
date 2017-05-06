from __future__ import print_function
import urllib2
import json
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import imghdr
import mark_girl as mg
import csv
import girl

GIRL_URL = "https://meizi.leanapp.cn/category/All/page/"
IMG_DIR = os.path.join(os.getcwd(), 'Images/')
TARGET_SIZE = (100, 100)
GIRL_MARK_FILE = "girl_img_mark.csv"


def download_girl_images(pagerange):
    for page in pagerange:
        data = json.load(urllib2.urlopen(GIRL_URL + str(page)))
        results = data['results']
        for girl in results:
            image_url = girl['image_url']
            download_girl_image(image_url)


def download_girl_image(image_url):
    image_name = image_url.split('/')[-1]
    f = urllib2.urlopen(image_url)
    data = f.read()
    s = IMG_DIR + image_name
    with open(s, "w") as girl:
        girl.write(data)
    return s


def process_image(image_filename):
    img = Image.open(image_filename)
    width, height = img.size
    side = min([width, height])
    left = (width - side) / 2
    upper = (height - side) / 2
    right = left + side
    bottom = upper + side
    area = (left, upper, right, bottom)
    img = img.crop(area).convert('RGB')
    img.thumbnail(TARGET_SIZE, Image.ANTIALIAS)
    img.save(image_filename)
    return img


def load_dataset():
    csvfile = open(GIRL_MARK_FILE, "r")
    reader = csv.reader(csvfile)
    data_arr = []
    for item in reader:
        image_filename = download_girl_image(item[0])
        if imghdr.what(image_filename) is not None:
            img = process_image(image_filename)
            data_arr.append(mg.mark_girl(img, int(item[1])))
    writer = tf.python_io.TFRecordWriter(mg.FILE_NAME)
    for data in data_arr:
        writer.write(data)
    writer.close()
    image_full, label_full = mg.load_mark_data()
    len_full = len(label_full)
    len_train = tf.cast(tf.ceil(len_full * 0.8), tf.int32)
    len_test = len_full - len_train
    image_train, image_test = tf.split(image_full, [len_train, len_test], 0)
    label_train, label_test = tf.split(label_full, [len_train, len_test], 0)
    train = girl.DataSet(image_train, label_train)
    test = girl.DataSet(image_test, label_test)
    return train, test
