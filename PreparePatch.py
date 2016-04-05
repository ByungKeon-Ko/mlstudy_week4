# 1. Purpose
#       load images in rect-#.txt list, generating pyramid images and counting # of 30x30 patches
#       and save them in "patches" folder
#       Also, record each original image name and location of patches in pyramid_describ.txt
#       so that we can compare with labels while training
# 2. output files
#       pyramid-#.jpg : pyramid images
#       pyramid_describ.txt : describe how many patches in each pyramid images, and which is original images
#           <pyramid_#.jpg>
#               <original image.jpg> <# of patches in this pyramid> <which one is face?>
#           * <which one is face?> should be interpreted in binary code
#               e.g) 0, 3th patches are face --> 2**0 + 2**3 = 1 + 8 = 9
#              --> This number is too big so change to record only face location
#       pyramid_index.txt
#           <# of patch> <scale> <left_x> <top_y> <width> <height>

import Image
import numpy as np
import math

from skimage.transform import pyramid_gaussian

# ------ Parameters ---------------------- 
base_path = "/home/ccrspeed/ml_study/week6/origin_data"
patch_path = "%s/../patches" % base_path
fold_num = 8

# ------ functions ------------------------
def write_contents () :
    global cnt_patch
    global patchNum
    global rect_loc
    global pyramids

    # print "write_contents!"
    for i in range( len(pyramids) ) :
        # describ_file.write ("pyramid-%s " %(i + cnt_patch) )
        # describ_file.write ("%d \n" %patchNum[i] )
        index_file.write("%d " %patchNum[i] )       # num of patches
        index_file.write("%d " %i)                  # scale
        index_file.write("%s \n" %rect_loc)         # rectangle location
    cnt_patch = cnt_patch + len(pyramids)

def save_pyramid () :
    global temp_line
    global pyramids
    global patchNum
    global total_patch
    global total_pyramid

    org_img = Image.open("%s/%s.jpg" %(base_path, temp_line), 'r' )
    
    org_img_name = "%s " %(temp_line)        # original image name
    # describ_file.write ( "%s\n" %org_img_name )  # original image name
    
    pyramids = list( pyramid_gaussian(org_img, downscale=math.sqrt(2) ) )
    for i in range(len(pyramids) ):
        if min( pyramids[i].shape[0], pyramids[i].shape[1] ) < 30 :
            del pyramids[i:]
            break
    
    for i in range( len (pyramids) ) :
        row = pyramids[i].shape[0]
        col = pyramids[i].shape[1]
        im_matrix = np.zeros([row, col, 3]).astype('uint8')
    
        for k in range(row):
            for j in range(col):
                im_matrix[k,j] = pyramids[i][k,j] * 255
    
        new_img = Image.fromarray(im_matrix)
          # new_img.save("%s/pyramid-%s.jpg" %(patch_path, i+total_patch) )
        new_img.save("%s/pyramid-%s.jpg" %(patch_path, i+total_pyramid) )
        # new_img.show()
    
        patchNum[i] = (row-30+1) * (col-30+1)                  # the number of patches
    total_pyramid = total_pyramid + len(pyramids)
    total_patch = total_patch + sum(patchNum)

# ----------------------------------------- 
# describ_file = open("%s/pyramid_describ.txt" %patch_path, 'w')
index_file   = open("%s/pyramid_index.txt"   %patch_path, 'w')

total_patch = 0
total_pyramid = 0

for i in range(1,fold_num+1) :
    rect_file_path  = "%s/FDDB-folds/rect-%s.txt"      %(base_path, i)
    rect_file  = open(rect_file_path,  'r')

    cnt_patch = 0

    patchNum = np.zeros( [10] ).astype('uint32')

    rect_loc = 0
    cnt_img = 0
    while 1 :
        if cnt_img%10 == 0 :
            print "current img : ", total_pyramid, cnt_img, total_patch
        temp_line = rect_file.readline().rstrip()

        cond_eof = temp_line == ''
        cond_numface = len(temp_line) <= 2
        if cond_numface != 1 :
            cond_newimg = (temp_line[0:3] == '200') & (temp_line[3]!=' ') & (temp_line[3]!='.')
        else :
            cond_newimg = 0

        if cond_eof :       # end of file
            # index_file.write("eof part!! ")
            write_contents()
            print "1 set done!!", i
            # index_file.write("eof done!! ")
            break

        elif cond_numface : # the number of face in the image
            pass
            # describ_file.write ("%s\n" %temp_line )

        elif cond_newimg :  # new image name
            if ( cnt_img != 0 ) :   # at the first case
                write_contents()

            cnt_img = cnt_img + 1
            save_pyramid()

        else :              # face location
            rect_loc = temp_line
            # describ_file.write ("%s\n" %temp_line )

# describ_file.write ("totalsum:%d" %total_patch )
index_file.write ("totalsum:%d" %total_patch )
print "index writing done!!"


