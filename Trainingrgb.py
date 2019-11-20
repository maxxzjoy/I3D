# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Loads a sample video and classifies using a trained Kinetics checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import i3d
import os

#import time

_SAMPLE_VIDEO_FRAMES = 16
_IMAGE_SIZE = 224
_BATCH_SIZE = 4

_TEST_SIZE = 1847
_TRAIN_SIZE = 9727   
_TRAIN_EPOCH = 1000

labelmask = np.zeros([12,12]).astype(np.int16)

train_epoch_iteration = int (_TRAIN_SIZE / _BATCH_SIZE)
test_epoch_iteration = int (_TEST_SIZE / _BATCH_SIZE)
iteration = train_epoch_iteration * _TRAIN_EPOCH

tfrecords_filename_train = '20191111trainingDay_9727_16.tfrecords'  # TFRecords 檔案名稱
tfrecords_filename_test = '20191111validDay_1847_16.tfrecords'  # TFRecords 檔案名稱
model_save_dir = './modelsDay2'

_SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}

_CHECKPOINT_PATHS = {
    'rgb': 'pretrainmodel/rgb_model-38240',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
    'rgb_NTU': 'modelsTWCC20191030/rgb_model-297680'
}

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'rgb', 'rgb, rgb600, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')

def read_and_decode(tfrecords_filename):
    # 建立檔名佇列
    filename_queue = tf.train.string_input_producer([tfrecords_filename]) 
    # 建立 TFRecordReader
    reader = tf.TFRecordReader()  # 建立 TFRecordReader
    # 讀取 TFRecords 的資料
    _, serialized_example = reader.read(filename_queue)  
    # 讀取一筆 Example
    features = tf.parse_single_example(serialized_example,
                                       features={
                                               'image_string': tf.FixedLenFeature([], tf.string),
                                               'label': tf.FixedLenFeature([], tf.float32)
                                               })
    # 將序列化的圖片轉為 uint8 的 tensor
    rgb_left_image = tf.decode_raw(features['image_string'], tf.float16)
    # 將 label 的資料轉為 float32 的 tensor
    label = tf.cast(features['label'], tf.float32)  
    # 將圖片調整成正確的尺寸
    rgb_left_image = tf.reshape(rgb_left_image, [_SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3])
    
    return rgb_left_image, label

def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  eval_type = FLAGS.eval_type
  imagenet_pretrained = FLAGS.imagenet_pretrained

  NUM_CLASSES = 12

  if eval_type not in ['rgb', 'rgb600', 'flow', 'joint']:
    raise ValueError('Bad `eval_type`, must be one of rgb, rgb600, flow, joint')


####################################################################
#build network varibles and architecture

  labels_placeholder = tf.placeholder(tf.float32, [_BATCH_SIZE, NUM_CLASSES])
  keep_prob = tf.placeholder(tf.float32)

  if eval_type in ['rgb', 'rgb600', 'joint']:
    # RGB input has 3 channels.
    rgb_input = tf.placeholder(
        tf.float32,
        shape=(_BATCH_SIZE, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))

    with tf.variable_scope('RGB'):
      rgb_model = i3d.InceptionI3d(
          NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      rgb_logits, _ = rgb_model(
          rgb_input, is_training=True, dropout_keep_prob=keep_prob)

    rgb_variable_map = {}
    rgb_saver_savedata = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'RGB':
          rgb_variable_map[variable.name.replace(':0', '')] = variable
          
    rgb_saver_savedata =  tf.train.Saver(var_list=rgb_variable_map, reshape=True)   #save model
#    del rgb_variable_map['RGB/inception_i3d/Logits/Conv3d_0c_1x1/conv_3d/w']
#    del rgb_variable_map['RGB/inception_i3d/Logits/Conv3d_0c_1x1/conv_3d/b']
#    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True) #load model


  model_logits = rgb_logits
  model_predictions = tf.nn.softmax(model_logits)
  output_class = tf.argmax(model_predictions, 1)
  
  # Read train and valid images TFRecords
  train_rgb_image, train_label = read_and_decode(tfrecords_filename_train) 
  test_rgb_image, test_label = read_and_decode(tfrecords_filename_test) 

  
  rgb_images, labels = tf.train.shuffle_batch([train_rgb_image, train_label],
                                            batch_size=_BATCH_SIZE,
                                            capacity=100,
                                            num_threads=1,
                                            min_after_dequeue=50)
  
  print('shuffle finished')
  
  label_batch_int = tf.cast(labels, tf.int32)
  label_batch = tf.reshape(tf.one_hot(label_batch_int, NUM_CLASSES), [-1, NUM_CLASSES])
    
  test_train_rgb_images, test_train_labels = tf.train.batch([train_rgb_image, train_label],
                                                        batch_size=_BATCH_SIZE,
                                                        num_threads=1,
                                                        capacity=25)
  
  test_train_label_batch_int = tf.cast(test_train_labels, tf.int32)
  test_train_label_batch = tf.reshape(tf.one_hot(test_train_label_batch_int, NUM_CLASSES), [-1, NUM_CLASSES])
    
  
  test_rgb_images, test_labels = tf.train.batch([test_rgb_image, test_label],
                                            batch_size=_BATCH_SIZE,
                                            num_threads=1,
                                            capacity=25)
  
  label_test_batch_int = tf.cast(test_labels, tf.int32)
  label_test_batch = tf.reshape(tf.one_hot(label_test_batch_int, NUM_CLASSES), [-1, NUM_CLASSES])
  
  
  global_step = tf.Variable(0)
  initial_learning_rate = 1e-5
  learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, 5000, decay_rate=0.9)

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels_placeholder, logits = model_logits))
  correct_prediction = tf.equal(tf.argmax(model_predictions, 1), tf.argmax(labels_placeholder, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#  train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)   #optimizer
  
  update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_op):
      train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)   #tf.identity(model_logits, name='train_op')

  init = tf.group(tf.global_variables_initializer(), 
                tf.local_variables_initializer())
  
  
  with tf.Session() as sess:
    
    sess.run(init)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    feed_dict = {}
    if eval_type in ['rgb', 'rgb600', 'joint']:
      if imagenet_pretrained:
        rgb_saver_savedata.restore(sess, _CHECKPOINT_PATHS['rgb_NTU'])
      else:
        rgb_saver_savedata.restore(sess, _CHECKPOINT_PATHS['rgb'])
      tf.logging.info('RGB checkpoint restored')
#      rgb_sample = np.load(_SAMPLE_PATHS['rgb'])
#      tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
      feed_dict[rgb_input] = np.zeros((_BATCH_SIZE, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3), dtype = np.float32)

    tmp = train_epoch_iteration * 2
    epoch_counter = 0
    
    print('Create file')
    file_output_train = open('file_output_train.txt', 'w')
    file_output_valid = open('file_output_valid.txt', 'w')
    
    for i in range(iteration):
#        start_time = time.time()
        get_train_rgb_images, get_train_labels = sess.run([rgb_images,label_batch])
        
        train_rgb_images = np.array(get_train_rgb_images).astype(np.float32)
        train_labels = np.array(get_train_labels).astype(np.float32)
        
        
        sess.run(train_op, feed_dict={
                rgb_input: train_rgb_images,
                labels_placeholder: train_labels,
                keep_prob: 0.5})

        if (i % tmp == 0):   
            
            '''testing train set'''
            test_loss_sum = 0.0
            test_accuracy_sum = 0.0
            for j in range(train_epoch_iteration):
                get_test_train_rgb_images, get_test_train_labels = sess.run([test_train_rgb_images, test_train_label_batch])
                test_train_rgb_images_data = np.array(get_test_train_rgb_images).astype(np.float32)
                
                test_train_labels_data = np.array(get_test_train_labels).astype(np.float32)
                test_loss, test_accuracy = sess.run([cross_entropy, accuracy], 
                                                    feed_dict={rgb_input: test_train_rgb_images_data,
                                                               labels_placeholder: test_train_labels_data,
                                                               keep_prob: 1})
                test_loss_sum = test_loss_sum + test_loss
                test_accuracy_sum = test_accuracy_sum + test_accuracy
                
            test_loss_average = test_loss_sum / train_epoch_iteration
            test_accuracy_average = test_accuracy_sum / train_epoch_iteration
            
            print('epoch:{}  train_lose:{:.5f}  train_accuracy:{:.5f}'.format(epoch_counter, test_loss_average, test_accuracy_average) )
            file_output_train.write(str(epoch_counter) + '\t,' + str(test_loss_average) +
                                    '\t,' + str(test_accuracy_average) + '\n')
            
            '''testing test set'''
            test_accuracy_sum = 0.0
            for k in range(test_epoch_iteration):
                get_test_rgb_images, get_test_labels = sess.run([test_rgb_images, label_test_batch])
                test_rgb_images_data = np.array(get_test_rgb_images).astype(np.float32)
                
                test_labels_data = np.array(get_test_labels).astype(np.float32)
#                test_loss, test_accuracy = sess.run([cross_entropy, accuracy], 
#                                                    feed_dict={rgb_input: test_rgb_images_data,
#                                                               labels_placeholder: test_labels_data,
#                                                               keep_prob: 1})
    
                test_loss, test_accuracy, predict = sess.run([cross_entropy, accuracy, output_class], 
                                                    feed_dict={rgb_input: test_rgb_images_data,
                                                               labels_placeholder: test_labels_data,
                                                               keep_prob: 1})
                label_arg = np.argmax(test_labels_data, 1)
                test_loss_sum = test_loss_sum + test_loss
                test_accuracy_sum = test_accuracy_sum + test_accuracy
                
                labelmask[label_arg[0], predict[0]] = labelmask[label_arg[0], predict[0]] + 1
                labelmask[label_arg[1], predict[1]] = labelmask[label_arg[1], predict[1]] + 1
                
            test_loss_average = test_loss_sum / test_epoch_iteration
            test_accuracy_average = test_accuracy_sum / test_epoch_iteration
            
            print('epoch:{}  test_lose:{:.5f}  test_accuracy:{:.5f}'.format(epoch_counter, test_loss_average, test_accuracy_average) )
            get_learning_rate = sess.run([learning_rate])
            print('learning rate:', get_learning_rate)
            file_output_valid.write(str(epoch_counter) + '\t,' + str(test_loss_average) + '\t,' +
                                    str(test_accuracy_average) + '\t,' + str(get_learning_rate) + '\n')
            print(labelmask)
            epoch_counter = epoch_counter + 1
            rgb_saver_savedata.save(sess, os.path.join(model_save_dir, './rgb_model'), i, write_state = False) 
            

#        duration = time.time() - start_time
#        print('Step %d: %.3f min' % (i, duration/60))
    coord.request_stop()
    coord.join(threads)
    file_output_train.close()
    file_output_valid.close()
    
if __name__ == '__main__':
  tf.app.run(main)
