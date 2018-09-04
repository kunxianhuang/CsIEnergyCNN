#!/bin/env python3
#-*-coding=utf-8 -*-
#Training for the CNN model
#written by: Chia-Te Chen (Apr., 2018)
#edited by:Kunxian Huang (May, 2018)

import os, sys
import glob, random
import time
import tensorflow as tf
from tensorflow.python.framework import graph_util
from CNNmodel import CNNmodel

DATA_PATH = 'train_data/'
BATCH_SIZE = 500
N_FEATURES = 900 #30*30 channnels, 351
n_steps = 30
pb_file_path = "save_model/pbfile/cnnmodel.pb"
def batch_generator(filenames, batchSize):
    """ 
    filenames is the list of files you want to read from.
    """
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TextLineReader(skip_header_lines=1) # skip the first line in the file
    _, value = reader.read(filename_queue)
    
    # record_defaults are the default values in case some of our columns are empty
    record_defaults = [[0.0] for _ in range(N_FEATURES)]
    record_defaults.append([0.0])
    record_defaults.append([0.0])

    # read in csv data
    content = tf.decode_csv(value, record_defaults=record_defaults) 

    for i in range(N_FEATURES):
      content[i] = content[i]
    
    content[-1] = content[-2]/content[-1] #Total E_{dep}/E_{true}

    # pack all 900 channel data into a tensor
    features = tf.stack(content[:N_FEATURES])

    # assign the last column to label
    label = tf.stack([content[-1]])

    # minimum number elements in the queue after a dequeue, used to ensure 
    # that the samples are sufficiently mixed
    # I think 10 times the BATCH_SIZE is sufficient
    min_after_dequeue = 0

    # the maximum number of elements in the queue (1 file for max 100,000 events)
    capacity = 200 * batchSize

    # shuffle the data to generate BATCH_SIZE sample pairs
    data_batch, label_batch = tf.train.shuffle_batch([features, label], batch_size=batchSize, 
                                        capacity=capacity, min_after_dequeue=min_after_dequeue)

    return data_batch, label_batch



sess = tf.InteractiveSession()

cnnmodel = CNNmodel()

#list for training files
filelist = glob.glob(DATA_PATH+"CsIArray_train*.csv")
#filelist = [os.path.basename(xfile) for xfile in filelist]

x = tf.placeholder(tf.float32, [BATCH_SIZE, N_FEATURES], name="input")
y_ = tf.placeholder(tf.float32, [BATCH_SIZE, 1], name="label")
keep_prob = tf.placeholder_with_default(1.0,shape=(), name="keep_prob") #tf.placeholder(tf.float32)

x_image = tf.reshape(x, [BATCH_SIZE, 30, 30, 1])
print(sess.run(tf.shape(x_image)))

training =  tf.placeholder_with_default(False,shape=(),name='training')

y_predict = cnnmodel.model(x_image, sess=sess, keep_prob=keep_prob, reuse_weight=False, is_training=True)

#loss and result
mse, error = cnnmodel.loss(y_predict, y_)
#result = cnnmodel.result(y_predict)

#optimizers
train_step = cnnmodel.optimizer_step1(mse, learn_rate=1e-3)
train_step_2 = cnnmodel.optimizer_step2(mse, learn_rate=1e-4)

# log of tensorflow
loss_summary = tf.summary.scalar("MSE", mse)
file_writer = tf.summary.FileWriter('save_model/',tf.get_default_graph())

#"StartTraining"
sess.run(tf.global_variables_initializer())

#save batch normalization moving mean and deviation variables    
tvars = tf.global_variables()
batchnorm_vars = [var for var in tvars if 'moving' in var.name]

save_vars = tf.trainable_variables() + batchnorm_vars
vnames = [var.name for var in save_vars]
print("All saving variables:{}".format(vnames))

#prepare to save model
saver = tf.train.Saver(save_vars)


train_mse_sum = 0.0
train_abserror_sum = 0.0
count = 0

test_mse_sum = 0.0
test_abserror_sum = 0.0

test_mse = 0.0
test_abserror = 0.0

nfiles = len(filelist)

epoch_mse_sum=0.0
epoch_abs_error_sum = 0.0
random.shuffle(filelist) # shuffle readin files every epoch
n_batches_file = 100000/BATCH_SIZE
#kfile=0
#fname = DATA_PATH+filelist[kfile]
data_batch, label_batch = batch_generator(filelist, BATCH_SIZE)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)
start_time = time.time()
for i in range(n_steps):
    #print("Step %d.....\r"%(i))
    train_time = (time.time()-start_time)/60.0 #in unit of mins
    sys.stdout.write("training time:%.2f min, Step: %05d \t %3.1f%%   \r" % (train_time, i, (1.0*i/n_steps)*100.0)) 
    sys.stdout.flush()
    
    #each epoch
    if (i%(n_batches_file*nfiles)==0 and i>0):
        #one epoch is finished
        #stop the previous thread
        
        iepoch = i/n_batches_file/nfiles
        print("Epoch %d\n \ttrain_sum_of_square: %g \t train_abserror: %g" %(iepoch, epoch_mse_sum/nfiles, epoch_abs_error_sum/nfiles))
        epoch_mse_sum=0.0
        epoch_abs_error_sum = 0.0
        
    features, labels  = sess.run([data_batch, label_batch])
    
    #print(features)
    #print("shape of labels")
    #print(sess.run(tf.shape(labels)))
    #print(sess.run(tf.shape(features)))
        
    train_mse = mse.eval(feed_dict={x:features, y_:labels, keep_prob: 1.0, training:True})
    train_abserror = error.eval(feed_dict={x:features, y_:labels, keep_prob: 1.0, training:True})
    
    train_mse_sum += train_mse
    train_abserror_sum += train_abserror

    epoch_mse_sum += train_mse
    epoch_abs_error_sum += train_abserror
    
    count += 1 #count for each steps
            
    if i%500 == 10:
        save_fn = "save_model/CsIArray_cnn_epoch%05d.ckpt" %(i)
        save_path = saver.save(sess, save_fn)
        print("------------step %d, average_sum_of_square %g"%(i, train_mse_sum/count))
        print("------------step %d, average_abserror %g"%(i, train_abserror_sum/count))
        
        y_predict_show = y_predict.eval(feed_dict={x:features, y_:labels, keep_prob: 1.0, training:True})
        loss_str = loss_summary.eval(feed_dict={x:features, y_:labels, keep_prob: 1.0, training:True})
        file_writer.add_summary(loss_str, i)
        for j in range(10):
            print("Predict ratio:{} \t True ratio: {}".format(y_predict_show[j][0], labels[j][0]))


    
    train_mse_sum = 0.0
    train_abserror_sum = 0.0
    count = 0

    if i <= 5000:
        train_step.run(feed_dict={x:features, y_:labels, keep_prob: 0.70, training:True})
    elif i <= 12000:
        train_step.run(feed_dict={x:features, y_:labels, keep_prob: 0.80, training:True})
    elif i <= 25000:
        train_step_2.run(feed_dict={x:features, y_:labels, keep_prob: 1.0, training:True})
    elif i <= 50000:
      train_step_2.run(feed_dict={x:features, y_:labels, keep_prob: 1.0, training:True})

      
coord.request_stop()
coord.join(threads)
    
file_writer.close()
# saving model as pb file
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["cnn_inception/output", "label", "input","mse","abserr","training"])
with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
    f.write(constant_graph.SerializeToString())
    
save_path = saver.save(sess, "save_model/CsIArray_cnn_final.ckpt")
print("Model saved in file: %s" % save_path)

sess.close()
