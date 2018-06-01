#!/bin/env python3
#*-*code=utf-8*-*

import tensorflow as tf
import numpy as np
import argparse
from CNNmodel import CNNmodel

DATA_PATH = 'test_data/'
SAVE_PATH = 'test_result/'
BATCH_SIZE = 500 #input test sample batch size
N_FEATURES = 900

def batch_generator(filenames, batchSize):
    """ filenames is the list of files you want to read from.
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
    
    content[-1] = content[-2]/content[-1]

    # pack all 900 features into a tensor
    features = tf.stack(content[:N_FEATURES])

    # assign the last column (Deposited energy/True energy) to label
    label = tf.stack([content[-1]])

    obsenergy = tf.stack([content[-2]])

    # minimum number elements in the queue after a dequeue, used to ensure 
    # that the samples are sufficiently mixed
    min_after_dequeue = 1

    # the maximum number of elements in the queue (1 file for max 100,000 events)
    capacity = 50000 * batchSize

    # shuffle the data to generate BATCH_SIZE sample pairs
    data_batch, label_batch, obs_ene = tf.train.batch([features, label, obsenergy], batch_size=batchSize, 
                                                      capacity=capacity)
    
    return data_batch, label_batch, obs_ene


parser = argparse.ArgumentParser(prog='CsIArray_CNN_apply.py')
parser.add_argument('--test_file',type=str,dest="testf",default='CsIArray_test_200_select.csv')
parser.add_argument('--test_outfile',type=str,dest="testout",default='CsIArray_testout_200.csv')

args = parser.parse_args()

#session 
sess= tf.InteractiveSession()

cnnmodel = CNNmodel()

x = tf.placeholder(tf.float32, [BATCH_SIZE, N_FEATURES])
y_ = tf.placeholder(tf.float32, [BATCH_SIZE, 1])

#keep_prob for drop out
keep_prob = tf.placeholder_with_default(1.0, shape=())

#training variable for batch norm
training = tf.placeholder_with_default(False,shape=(),name='training')

x_image = tf.reshape(x, [BATCH_SIZE, 30, 30, 1])
#print(sess.run(tf.shape(x_image)))

print("Shape of input images {}.".format(sess.run(tf.shape(x_image))))
y_predict = cnnmodel.model(x_image,  sess=sess, keep_prob=keep_prob, reuse_weight=False, is_training=training)

#loss and result
mse, error = cnnmodel.loss(y_predict, y_)


saver = tf.train.Saver()
save_final_fn ="save_model/CsIArray_cnn_final.ckpt"
#"save_model/train_0522/CsIArray_cnn_final.ckpt" 
saver.restore(sess, save_final_fn)
print("Model restore from file: %s" %(save_final_fn))

test_fn = DATA_PATH + args.testf

#with open(test_fn) as rf:
#    DATA_SIZE = sum(1 for line in rf) -1 # count lines of the test file
DATA_SIZE=int(45000/BATCH_SIZE)

data_batch, label_batch, obs_ene = batch_generator([test_fn], BATCH_SIZE)


coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

test_mse_sum = 0.0
test_abserror_sum = 0.0
number_count = 0

save_fn = SAVE_PATH+args.testout
f=open(save_fn,'w')

print("Evaluate data size of {}.".format(DATA_SIZE))


#print out restore variables
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
#print_tensors_in_checkpoint_file(file_name=save_final_fn, tensor_name='', all_tensors=False, all_tensor_names=True)


f.write(str("Sum,ratio_predict,Ene_predict,ratio_true,TrueEnergy\n"))
for i in range(DATA_SIZE):
    
    features, labels, Totenergy  = sess.run([data_batch, label_batch, obs_ene])

    feed_dict={x:features, y_:labels, keep_prob:1.0,training:False}

    test_mse = mse.eval(feed_dict=feed_dict)
    test_mse_sum = test_mse_sum + np.sum(test_mse)
    test_abserror = error.eval(feed_dict=feed_dict)
    test_abserror_sum = test_abserror_sum + np.sum(test_abserror)
    
    output_result = y_predict.eval(feed_dict=feed_dict)
    
    #remove single-dimensional entries
    labels = np.squeeze(labels)
    Totenergy = np.squeeze(Totenergy)
    output_result = np.squeeze(output_result)
    for iis in range(BATCH_SIZE):
        f.write(str(Totenergy[iis]) + ",")
        f.write(str(output_result[iis]) + ",")
        f.write(str(Totenergy[iis]/output_result[iis]) + ",")
        f.write(str(labels[iis]) + ",")
        f.write(str(Totenergy[iis]/labels[iis]) + "\n")
    
    number_count = number_count + 1
    
    
    if i%1000 == 0:
        print("test_sum_of_square %g, test_abserror %g"%(test_mse_sum/(number_count*BATCH_SIZE) , test_abserror_sum/(number_count*BATCH_SIZE)))
        
f.close()

coord.request_stop()
coord.join(threads)

sess.close()
