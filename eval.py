#
#
#

import Image
import numpy as np
import tensorflow as tf
import random
import math

from skimage.transform import pyramid_gaussian

# ------ Parameters ---------------------- 
base_path = "/home/ccrspeed/ml_study/week6/origin_data"
# test_file_name = "2002/08/11/big/img_591"
# output_name = "591"

# test_file_name = "2002/07/22/big/img_43"
# output_name = "43"
# test_file_name = "2003/01/15/big/img_884"
# output_name = "884"
# test_file_name = "2002/08/22/big/img_729"
# output_name = "729"

windowSize = 3
colorNum = 3
nBatch = 100
threshold = 0.95

test_list = "/home/ccrspeed/ml_study/week6/origin_data/FDDB-folds/rect-9.txt"

BigKernelSize = 100

# ------ functions ------------------------

# ----------------------------------------- 

# ---- Network modeling ------------------------------------------------------------------- #
sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.random_normal(shape, stddev=0.01, name='initial')
    # initial = tf.truncated_normal(shape, stddev=0.01, name='initial')
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W) :
    # return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

x = tf.placeholder( tf.float32, [None, BigKernelSize*BigKernelSize*3], name = 'x' )
y_ = tf.placeholder(tf.float32, [None , 1], name = 'y_' )

# ----- 1st Convolutional Layer --------- #
# input size  : len   x len   x 3  channel ( 4D : nBatch, col, row, features(color) )
# output size : len/2 x len/2 x 32 channel ( 4D : nBatch, col/2, row/2, features(32) )
KernelSize = 3

W_conv1 = weight_variable([KernelSize, KernelSize, colorNum, 32] ) # input channels : 1, output channels : 32
# W_conv1 = weight_variable([KernelSize, KernelSize, 1, 32] ) # input channels : 1, output channels : 32
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,BigKernelSize,BigKernelSize, colorNum], name='x_image')
temp1 = conv2d(x_image, W_conv1)

h_conv1 = tf.nn.relu( temp1 + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# ---- 2nd Convolutional Layer ---------- #
# input  : len/2 x len/2 x 32 channel ( 4D : nBatch, col/2, row/2, features(32) )
# output : len/4 x len/4 x 32 channel ( 4D : nBatch, col/4, row/4, features(32) )

W_conv2 = weight_variable([KernelSize, KernelSize, 32,32] )
b_conv2 = bias_variable([32])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# ---- 3rd Convolutional Layer w/o pooling ---------- #
# input  : len/4 x len/4 x 32 channel ( 4D : nBatch, col/4, row/4, features(32) )
# output : len/4 x len/4 x 32 channel ( 4D : nBatch, col/4, row/4, features(32) )

W_conv3 = weight_variable([KernelSize, KernelSize, 32,32] )
b_conv3 = bias_variable([32])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3 )

# ----- 4th Fully Connected Layer ( FC layer ) --------- #
# input  : len/4 x len/4 x 32 ch ( 4D : nBatch, col/4, row/4, features(32) )
# output : len/4 x len/4 x 16 ch ( 4D : nBatch, col/4, row/4, features(16) )
windowSize4 = 4
W_fc1 = weight_variable([windowSize4*windowSize4*32, 256] )
b_fc1 = bias_variable([256])

# h_conv3_flat = tf.reshape(h_conv3, [-1, windowSize4*windowSize4*32] )
# temp_fc1 = tf.matmul(h_conv3_flat, W_fc1) + b_fc1 
W_fc1_reshape = tf.reshape(W_fc1, [windowSize4, windowSize4, 32, 256] )
W_fc1_t = tf.transpose(W_fc1_reshape, [0, 1, 2, 3] )
temp_fc1 = conv2d(h_conv3, W_fc1_t) + b_fc1

h_fc1 = tf.nn.relu( temp_fc1 )

# --- 5th FC Layer : Readout Layer --------------------- #
# input  : 256 channel ( 4D : nBatch, features(256 = 1/2 * 32*(col-14)/4*(row-14)/4 ) )
# output : 1 channel   ( 4D : nBatch, features(1   = 1/256 * 1/2 * 32*(col-14)/4*(row-14)/4 ) )
windowSize5 = 1
W_fc2 = weight_variable([256, 1] )
b_fc2 = bias_variable([1])

# temp_node4 = tf.matmul(h_fc1, W_fc2) + b_fc2
W_fc2_reshape = tf.reshape(W_fc2, [windowSize5, windowSize5, 256,1] )
W_fc2_t = tf.transpose(W_fc2_reshape, [0, 1, 2, 3] )
temp_node4 = conv2d(h_fc1, W_fc2_t) + b_fc2
y_conv = tf.sigmoid( temp_node4 )

# --- Train and Evaluate ------------------- #
saver = tf.train.Saver()

saver.restore(sess, "tmp/model.ckpt")
print "Model restored"

rect_file = open(test_list,  'r')

while 1 :
    temp_line = rect_file.readline().rstrip()
    
    cond_eof = temp_line == ''
    cond_numface = len(temp_line) <= 2
    if cond_numface != 1 :
        cond_newimg = (temp_line[0:3] == '200') & (temp_line[3]!=' ') & (temp_line[3]!='.')
    else :
        cond_newimg = 0

    if cond_eof :       # end of file
        print "1 set done!!"
        break
    
    elif cond_numface : # the number of face in the image
        pass
    
    elif cond_newimg :  # new image name
        org_img_file = Image.open("%s/%s.jpg" %(base_path, temp_line) )
        
        pyramids = list( pyramid_gaussian(org_img_file, downscale=math.sqrt(2) ) )
        # pyramids = list( pyramid_gaussian(org_img_file, downscale=2 ) )
        
        for i in range(len(pyramids) ):
            if min( pyramids[i].shape[0], pyramids[i].shape[1] ) < 30 :
            # if min( pyramids[i].shape[0], pyramids[i].shape[1] ) < BigKernelSize :
                del pyramids[i:]
                break
            
        cnt_face = 0
        
        contents = []
        # contents.append( test_file_name )
        contents.append( temp_line )
        
        batch = np.zeros([BigKernelSize,BigKernelSize,3])

        def next_batch(j, i) :
            global pos_y
            global pos_x
            global BigKernelSize
            for pos_y in range(BigKernelSize) :
                for pos_x in range(BigKernelSize) :
                    if (j + pos_y >= pyramids[l].shape[0]) | (i + pos_x >= pyramids[l].shape[1]) :
                        batch[pos_y, pos_x] = [0,0,0]
                    else :
                        batch[pos_y, pos_x] = pyramids[l][j + pos_y, i + pos_x]
        
            return batch
        
        for l in range( 2, len(pyramids) ) :
            temp_row = pyramids[l].shape[0]
            temp_col = pyramids[l].shape[1]
        
            print " ONE pyramid!! %d" %(l)
            j = 0
            i = 0

            while 1 :
                print " one loop!! %d, %d" %(l,j)
                while 1 :

                    batch = next_batch(j,i)

                    batch_reshape = np.reshape(batch, [1, BigKernelSize*BigKernelSize*3] )
                    y_val = sess.run(y_conv, feed_dict={x:batch_reshape } )
                    y_val = np.reshape(y_val, [(BigKernelSize-14)/4-3, (BigKernelSize-14)/4 -3] )
                    # y_val = np.reshape(y_val, [BigKernelSize/4, BigKernelSize/4 ] )

                    y_val_shape = np.shape(y_val)

                    for bigpatch_y in range(y_val_shape[0]):
                        for bigpatch_x in range(y_val_shape[1]):
                            if y_val[bigpatch_y,bigpatch_x] >= threshold :
                                # print "correct!!"
                                org_row = (j + bigpatch_y*4) * math.sqrt(2)**l
                                org_col = (i + bigpatch_x*4) * math.sqrt(2)**l
                                org_len = 30 * math.sqrt(2)**l
                                contents.append( "%d %d %d %d %f %d" %(org_col, org_row + org_len, org_len, org_len, y_val[bigpatch_y, bigpatch_x], l ) )
                                cnt_face = cnt_face + 1
                            else :
                                pass
                                # print "wrong!!"
        
                    if i + BigKernelSize-30 < temp_col :
                        i = i + (BigKernelSize-30)*4/9
                    else :
                        i = temp_col - BigKernelSize -1

                    if i < temp_col-(BigKernelSize-1) :
                        break
                    
                if j + BigKernelSize-30 < temp_row :
                    j = j + (BigKernelSize-30)*4/9
                else :
                    j = temp_row - BigKernelSize -1

                if j < temp_row-(BigKernelSize-1) :
                    break



        contents.insert(1, str(cnt_face) )
        
        output_file = open("%s/../output_file/out-%s.txt" %(base_path, temp_line.split('/')[4]), 'w' )
        
        for i in range(cnt_face+2) :
            output_file.write("%s\n" %contents[i] )


