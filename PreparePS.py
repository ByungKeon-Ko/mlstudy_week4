# 1. Purpose
#       find and generate positive sample of which size is 32x32
#

import Image
import numpy as np

# ------ Parameters ---------------------- 
base_path = "/home/ccrspeed/ml_study/week6/origin_data"
ps_path = "positive_sample"
rect_path = "FDDB-folds"
fold_num = 8

# ------ functions ------------------------
def load_orgimg(temp_line) :
    global org_img
    global base_path
    global org_width, org_height

    print "load_orgimg function!"
    org_file = Image.open("%s/%s.jpg" %(base_path, temp_line), 'r' )
    org_img = org_file.load()
    org_width, org_height = org_file.size

    return 1

def gen_ps(temp_line, total_ps) :
    global org_img

    print "gen_ps function!"
    temp_line = temp_line.split()
    left_x =  int( temp_line[0] )
    top_y  =  int( temp_line[1] )
    width  =  int( temp_line[2] )
    height =  int( temp_line[3] )

    center_x = left_x + width/2
    center_y = top_y - height/2
    right_x  = left_x + width
    btm_y    = top_y  - height

    # ps_len = max(width, height)
    ps_len = min(width, height)       # change maximum square face to minimum square
    if ps_len > min(org_width, org_height) :
        ps_len = min(org_width, org_height)

    #   if height > width :         # change maximum square face to minimum square
    #       ps_y = top_y - height
    #       ps_x = center_x - height/2
    #   else :
    #       ps_y = center_y - width/2
    #       ps_x = left_x

    ps_y = center_y -ps_len/2
    ps_x = center_x -ps_len/2

    # Overflow & Underflow index preventing
    if (ps_x + ps_len) >= org_width :
        ps_x = org_width - ps_len - 1
    if ps_y < 0 :
        ps_y = 0
    if (ps_y + ps_len) >= org_height :
        ps_y = org_height - ps_len - 1
    if ps_y < 0 :
        ps_y = 0
    if ps_x < 0 :
        ps_x = 0

    pos_smpl = np.zeros( [ps_len, ps_len, 3] ).astype('uint8')

    for j in range(ps_len) :
        for i in range(ps_len) :
            pos_smpl[j,i] = org_img[ i + ps_x, j + ps_y ]

    pos_img = Image.fromarray( pos_smpl )
    pos_img = pos_img.resize( (30,30), Image.BICUBIC )
    # pos_img.show()
    pos_img.save( "%s/ps-%d.jpg" %(ps_path, total_ps) )
    print "generate %dth-positive sample!" %total_ps




# ----------------------------------------- 

total_ps = 0    # total # of positive samples

print "PreparePS.py start!!"


for i in range(1, fold_num +1) :
    rect_file = open("%s/%s/rect-%s.txt" %(base_path, rect_path, i), 'r')

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
            load_orgimg(temp_line)

        else :              # face location
            gen_ps(temp_line, total_ps)
            total_ps = total_ps + 1

print "index writing done!!"
print "total_ps = ", total_ps




