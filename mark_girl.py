import tensorflow as tf
from PIL import Image
import fetch_girl_images as fgi
import os

FILE_NAME = "training-image.tfrecord"
MARK_IMG_DIR = os.path.join(os.getcwd(), 'MarkImages/')


def mark_girl(img, mark):
    img = img.resize(fgi.TARGET_SIZE)
    img_raw = img.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[mark])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    }))
    return example.SerializeToString()


def load_mark_data(num=1):
    filename_queue = tf.train.string_input_producer([FILE_NAME])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    image = tf.decode_raw(features['img_raw'], tf.uint8)
    size = fgi.TARGET_SIZE
    image = tf.reshape(image, [size[0], size[1], 3])
    label = tf.cast(features['label'], tf.int32)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(num):
            example, l = sess.run([image, label])
            img = Image.fromarray(example, 'RGB')
            img.save(MARK_IMG_DIR + str(i) + '_''Label_' + str(l) + '.jpg')
        coord.request_stop()
        coord.join(threads)
    return image, label

