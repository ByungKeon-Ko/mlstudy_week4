# 1. Purpose
#       Train the model.
#       Load pyramid images in patches folder and generate patches
#       Save weights
#
# 2. Program order
#       (1) ellipse2rect.py
#       (2) PreparePatch.py
#       (3) PreparePS.py
#       (4) train.py            <- you need to specify "psNum" which you can get by running PreparePS.py
#       (5) eval.py             <- you need to specify target file name that you want ( test_file_name, output_name )
#       (6) NonMaximum.py       <- you need to specify target file name that you want ( test_file_name )
#       (7) RedRectangle.py     <- you need to specify target file name that you want ( test_file_name )
#
# 3. TDO
#       change (5), (6) for running automatically for serveral images in fold9~10
# 

import Image
import numpy as np
import tensorflow as tf
import random
import PIL
from PIL import ImageOps

# ------ Parameters ---------------------- 
base_path = "/home/ccrspeed/ml_study/week6/origin_data"
patch_path = "%s/../patches" % base_path
# output_report_path = "loss_graph"
output_report_path = "loss_graph_neg"

ReEXU_MODE = 0

windowSize = 3
colorNum = 3
nBatch = 50
threshold = 0.99
if ReEXU_MODE :
    LearningRate = 0.0001
else :
    LearningRate = 0.0005
psNum = 4136             # # of positive samples

# ------ functions ------------------------
#   def face_diagnose(row, col, left_x, top_y, width, height, margin, scale ):
#       # actual scale : 2**scale
#   
#       for i in range(scale) :
#           left_x  = left_x/2
#           top_y   = top_y /2
#           width   = width /2
#           height  = height/2
#   
#       center_x = left_x + width/2
#       center_y = top_y - height/2
#       right_x  = left_x + width
#       btm_y    = top_y  - height
#   
#       cond1_left_x  = min ( left_x  - margin, center_x - height/2 )
#       cond1_right_x = max ( right_x + margin, center_x + height/2 )
#       cond1_top_y   = max ( top_y   + margin, center_y + width/2  )
#       cond1_btm_y   = min ( btm_y   - margin, center_y - width/2  )
#   
#       cond2_left_x  = max ( left_x  + margin, center_x - height/2 )
#       cond2_right_x = min ( right_x - margin, center_x + height/2 )
#       cond2_top_y   = min ( top_y   - margin, center_y + width/2  )
#       cond2_btm_y   = max ( btm_y   + margin, center_y - width/2  )
#   
#       x_cond1 = ( col >= cond1_left_x ) & ( col+30-1 <= cond1_right_x )
#       y_cond1 = ( row >= cond1_btm_y )  & ( row+30-1 <= cond1_top_y   )
#       x_cond2 = ( col <= cond2_left_x ) & ( col+30-1 >= cond2_right_x )
#       y_cond2 = ( row <= cond2_btm_y )  & ( row+30-1 >= cond2_top_y   )
#   
#       if x_cond1 & y_cond1 & x_cond2 & y_cond2 :
#           # print "true semple!!!"
#           return 1
#       else :
#           return 0

rand_index = 0
img_index = 0


class BatchManager ( ) :
    def init (self):
        self.index_file = open("%s/../patches/pyramid_index.txt" %patch_path, 'r')
        self.pyramid_index = self.index_file.readlines()
        for i in range( len(self.pyramid_index) ) :
            self.pyramid_index[i] = self.pyramid_index[i].rstrip()
        self.ns_max_index = int(self.pyramid_index[len(self.pyramid_index)-1].split(':')[1] )

        self.ps_max_index = psNum

    def next_batch (self, nBatch):
        x_batch = np.zeros([nBatch, 30, 30, 3]).astype('float32')
        y_batch = np.zeros([nBatch, 1]).astype('uint8')

        for i in range(nBatch) :
            if random.randint(0,1) :
                x_batch[i], y_batch[i] = self.ns_batch()
            else :
                x_batch[i], y_batch[i] = self.ps_batch()

        x_batch = np.reshape(x_batch, [nBatch, 30*30*3] ) + np.random.randn(nBatch, 30*30*3) * 4.0/255.0
        return [x_batch, y_batch]

    def ps_batch (self):
        x_batch = np.zeros([30, 30, 3]).astype('float32')
        y_batch = np.zeros([1]).astype('uint8')

        rand_index = random.randint(0, self.ps_max_index-1)

        org_file = Image.open("%s/../positive_sample/ps-%s.jpg" %(base_path, rand_index), 'r' )
        org_file = org_file.rotate(random.randint(-20, 20) )
        if random.randint(0,1) :
            org_file = PIL.ImageOps.mirror(org_file)
        org_img = org_file.load()

        for j in range(30) :
            for i in range(30) :
                x_batch[j,i] = [float(org_img[ i , j ][0])/255.0, float(org_img[ i , j ][1])/255.0, float(org_img[ i , j ][2])/255.0 ]

        y_batch = np.ones([1])
        return [x_batch, y_batch]

    def ns_batch (self):
        x_batch = np.zeros([30, 30, 3]).astype('float32')
        y_batch = np.zeros([1]).astype('uint8')
        global rand_index
        global img_index
        img_index = 0       # index for selecting a pyramid image

        rand_index = random.randint(0, self.ns_max_index -1 )    # index for a patch in a pyramid image
        # rand_index = 212530 + 500
        # pyramid_img_index = rand_index
        global patchNum

        for i in range(len(self.pyramid_index)-1 ):
            patchNum = int(self.pyramid_index[i].split()[0] )
            if ( rand_index >= patchNum ):
                rand_index = rand_index - patchNum
            else :
                img_index = i
                break
        # pyramid_img_index = pyramid_img_index - rand_index

        # --- X Batch ----------------- 
        global row, col, pyramid_img_inst, pyramid_img
        pyramid_img = Image.open("%s/pyramid-%s.jpg"%(patch_path, img_index ) )
        # pyramid_img = Image.open("%s/pyramid-%s.jpg"%(patch_path, pyramid_img_index) )
        # pyramid_img.show()
        col_max = pyramid_img.size[0]
        row_max = pyramid_img.size[1]
        pyramid_img = pyramid_img.rotate(random.randint(-20, 20) )
        if random.randint(0,1) :
            pyramid_img = PIL.ImageOps.mirror(pyramid_img)
        pyramid_img_inst = pyramid_img.load()
        
        row = 0
        col = rand_index
        
        while ( col >= col_max - 30 +1 ) :
            row = row + 1
            col = col - (col_max-30+1)
        
        for j in range(30):
            for i in range(30):
                x_batch[j,i] = [float(pyramid_img_inst[i+col, j+row][0])/255.0, float(pyramid_img_inst[i+col, j+row][1])/255.0, float(pyramid_img_inst[i+col, j+row][2])/255.0 ]
                # x_batch[k,j,i] = [10, 255, 0]

        y_batch = np.zeros([1])
        return [x_batch, y_batch]


# ----------------------------------------- 

# ---- Network modeling ------------------------------------------------------------------- #
sess = tf.InteractiveSession()

def weight_variable(shape):
    # initial = tf.random_normal(shape, stddev=0.01, name='initial')
    initial = tf.truncated_normal(shape, stddev=0.1, name='initial')
    return tf.Variable(initial)

def bias_variable(shape):
    # initial = tf.constant(0.1, shape=shape)
    initial = tf.truncated_normal(shape, mean=0.1, stddev=0.05, name='initial')
    # initial = tf.truncated_normal(shape, mean=-0.1, stddev=0.05, name='initial')
    # initial = tf.truncated_normal(shape, stddev=0.05, name='initial')
    return tf.Variable(initial)

def conv2d(x,W) :
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

def ReEXU(x):
    x_exp = tf.exp(x) -1
    x_result = tf.maximum(0.0, x_exp )
    return x_result

x = tf.placeholder( tf.float32, [None, 30*30*3], name = 'x' )
y_ = tf.placeholder(tf.float32, [None , 1], name = 'y_' )

# ----- 1st Convolutional Layer --------- #
# input size  : 30x30 x 3  channel ( 4D : nBatch, col, row, features(color) )
# output size : 14x14 x 32 channel ( 4D : nBatch, (col-2)/2, (row-2)/2, features(32) )
windowSize1 = 3

W_conv1 = weight_variable([windowSize1, windowSize1, colorNum, 32] ) # input channels : 1, output channels : 32
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,30,30, colorNum], name='x_image')
# x_image = tf.image.random_flip_left_right(x_image)
# x_image = tf.image.random_brightness(x_image, max_delta=(63.0/255.0) )
# x_image = tf.image.random_contrast(x_image, lower=0.2, upper=1.8)
# x_image = tf.image.random_hue(x_image, max_delta=0.2 )

temp1 = conv2d(x_image, W_conv1)

if ReEXU_MODE :
    h_conv1 = ReEXU(temp1 + b_conv1 )
else :
    h_conv1 = tf.nn.relu( temp1 + b_conv1)

h_pool1 = max_pool_2x2(h_conv1)

# ---- 2nd Convolutional Layer ---------- #
# input : 14 x 14 x 32 channel ( 4D : nBatch, (col-2)/2, (row-2)/2, features(32) )
# output : 6 x 6 x 32 channel  ( 4D : nBatch, (col-6)/4, (row-6)/2, features(32) )
windowSize2 = 3

W_conv2 = weight_variable([windowSize2, windowSize2, 32,32] )
b_conv2 = bias_variable([32])

if ReEXU_MODE :
    h_conv2 = ReEXU(conv2d(h_pool1, W_conv2) + b_conv2)
else :
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

h_pool2 = max_pool_2x2(h_conv2)

# ---- 3rd Convolutional Layer w/o pooling ---------- #
# input  : 6 x 6 x 32 channel ( 4D : nBatch, (col-6)/4,  (row-6)/4,  features(32) )
# output : 4 x 4 x 32 channel ( 4D : nBatch, (col-14)/4, (row-14)/4, features(32) )
windowSize3 = 3

W_conv3 = weight_variable([windowSize3, windowSize3, 32,32] )
b_conv3 = bias_variable([32])

if ReEXU_MODE :
    h_conv3 = ReEXU(conv2d(h_pool2, W_conv3) + b_conv3 )
else :
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3 )

# ----- 4th Fully Connected Layer ( FC layer ) --------- #
# input : 4 x 4 x 32 channel ( 4D : nBatch, (col-14)/4, (row-14)/4, features(32) )
# output : 256 channel       ( 4D : nBatch, features(256 = 1/2 * 32*(col-14)/4*(row-14)/4 ) )
windowSize4 = 4
W_fc1 = weight_variable([windowSize4*windowSize4*32, 256] )
b_fc1 = bias_variable([256])

h_conv3_flat = tf.reshape(h_conv3, [-1, windowSize4*windowSize4*32] )

if ReEXU_MODE :
    h_fc1 = ReEXU(tf.matmul(h_conv3_flat, W_fc1) + b_fc1 )
else :
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1 )

#   # --- Dropout  --------------------------- #
#   keep_prob = tf.placeholder("float")
#   h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob )

# --- 5th FC Layer : Readout Layer --------------------- #
# input  : 256 channel ( 4D : nBatch, features(256 = 1/2 * 32*(col-14)/4*(row-14)/4 ) )
# output : 1 channel   ( 4D : nBatch, features(1   = 1/256 * 1/2 * 32*(col-14)/4*(row-14)/4 ) )
W_fc2 = weight_variable([256, 1] )
b_fc2 = bias_variable([1])

temp_node4 = tf.matmul(h_fc1, W_fc2) + b_fc2
y_conv = tf.sigmoid( temp_node4 )

# y_conv = tf.nn.softmax( temp_node3 )

# --- Train and Evaluate ------------------- #
cross_entropy = -tf.reduce_mean(y_*tf.log(y_conv) + (1-y_) *tf.log(1-y_conv) )
# cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv)  )
train_step = tf.train.AdamOptimizer(LearningRate).minimize(cross_entropy)

temp_node2 = tf.greater( y_conv, threshold )
temp_node = tf.cast( temp_node2, tf.float32)
correct_prediction = tf.equal( temp_node , y_ )
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float" ) )

saver = tf.train.Saver()
init = tf.initialize_all_variables()

sess.run( init )

BM = BatchManager()
BM.init()

learning_scale = 1

output_report = open(output_report_path, 'w')
cnt = 0

for i in range(1000):
    # if (i%200 == 0)&(i!=0) :
    #     nBatch = nBatch + 10

    batch = BM.next_batch(nBatch)
    if i%10 ==0 :
        loss = cross_entropy.eval(feed_dict={x:batch[0][0:nBatch], y_:batch[1][0:nBatch] } )
        print loss
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1] } )
        print "step %d, training accuracy %g" %(i, train_accuracy)
    if i%100 == 0 :
        print W_conv1.eval()[0,0,0]
        print b_conv1.eval()
    if loss < (0.1/learning_scale) :
        learning_scale = learning_scale*2
        nBatch = nBatch * learning_scale
        print "!!!!!!!!!! change learning scale : ", learning_scale
        # train_step = tf.train.AdamOptimizer(LearningRate).minimize(cross_entropy)

    train_step.run(feed_dict={x:batch[0], y_: batch[1] } )

    loss = cross_entropy.eval(feed_dict={x:batch[0][0:nBatch], y_:batch[1][0:nBatch] } )
    cnt = cnt + nBatch
    output_report.write("%s %s\n" %( cnt, loss) )

    if i%100 == 0 :
        # save_path = saver.save(sess, "tmp/model.ckpt")
        # save_path = saver.save(sess, "tmp/model_neg.ckpt")
        save_path = saver.save(sess, "tmp/model_w5.ckpt")
        # save_path = saver.save(sess, "tmp/model_reexu.ckpt")
        print "Model saved in file: ", save_path

# test_accuracy = accuracy.eval(feed_dict={x:mnist.test.images[0:100], y_: mnist.test.labels[0:100], keep_prob:1.0} )

# print "test accuracy %g"%test_accuracy
