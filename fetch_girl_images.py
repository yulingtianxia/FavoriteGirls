from __future__ import print_function
import urllib
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
IMG_TRAIN_DIR = os.path.join(os.getcwd(), 'ImagesTrain/')
IMG_TEST_DIR = os.path.join(os.getcwd(), 'ImagesTest/')
TARGET_SIZE = (128, 128)
GIRL_MARK_FILE = "girl_img_mark.csv"


def download_girl_images(pagerange):
    for page in pagerange:
        data = json.load(urllib.request.urlopen(GIRL_URL + str(page)))
        results = data['results']
        for girl in results:
            image_url = girl['image_url']
            download_girl_image(image_url)


def download_girl_image(image_url, des_dir):
    image_name = image_url.split('/')[-1]
    f = urllib.request.urlopen(image_url)
    data = f.read()
    s = des_dir + image_name
    with open(s, "wb") as girl:
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


def download_proprocess_dataset():
    csvfile = open(GIRL_MARK_FILE, "r")
    reader = csv.reader(csvfile)
    train_data_arr = []
    test_data_arr = []
    for index, item in enumerate(reader):
        if index % 5 == 0:
            image_filename = download_girl_image(item[0], IMG_TEST_DIR)
            if imghdr.what(image_filename) is not None:
                img = process_image(image_filename)
                test_data_arr.append(mg.mark_girl(img, int(item[1])))
        else:
            image_filename = download_girl_image(item[0], IMG_TRAIN_DIR)
            if imghdr.what(image_filename) is not None:
                img = process_image(image_filename)
                train_data_arr.append(mg.mark_girl(img, int(item[1])))
    writer = tf.python_io.TFRecordWriter(mg.FILE_NAME_TEST)
    for data in test_data_arr:
        writer.write(data)
    writer.close()
    writer = tf.python_io.TFRecordWriter(mg.FILE_NAME_TRAIN)
    for data in train_data_arr:
        writer.write(data)
    writer.close()


def load_dataset(train_len=320, test_len=80):
    image_train, label_train = mg.load_train_data()
    image_test, label_test = mg.load_test_data()

    image_train_full = []
    label_train_full = []
    image_test_full = []
    label_test_full = []

    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(train_len):
            x_train, y_train = sess.run([image_train, label_train])
            image_train_full.append(x_train)
            label_train_full.append(y_train)
        for i in range(test_len):
            x_test, y_test = sess.run([image_test, label_test])
            image_test_full.append(x_test)
            label_test_full.append(y_test)
        coord.request_stop()
        coord.join(threads)

    train = girl.DataSet(np.array(image_train_full), np.array(label_train_full))
    test = girl.DataSet(np.array(image_test_full), np.array(label_test_full))
    return train, test
