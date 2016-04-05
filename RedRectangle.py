# 1. Purpose
#      read rect-#.txt file and generate new images with red rectangles

import numpy as np
import Image

# ------ Parameters ---------------------- 
base_path = "/home/ccrspeed/ml_study/week6/origin_data"
text_path = "/home/ccrspeed/ml_study/week6/output_file_1"

new_path = "./output_image"
# target_file_name = "%s/../output_file_1/out-591.txt" %(base_path)
# target_file_name = "%s/../output_file_1/out-43.txt" %(base_path)
# target_file_name = "%s/../output_file_1/out-884.txt" %(base_path)
# target_file_name = "%s/../output_file_1/out-729.txt" %(base_path)

TEST_MODE = 1
if TEST_MODE :
    fold_num = 1
else :
    fold_num = 8

test_list = "/home/ccrspeed/ml_study/week6/origin_data/FDDB-folds/rect-9.txt"

# ------ functions ------------------------
def draw_rectangle (im_matrix, left_x, top_y, width, height) :
    row, col, colorNum = im_matrix.shape

    right_x = left_x + width
    btm_y = top_y - height

    if(right_x >= col) :
        right_x = col -1
    if left_x < 0 :
        left_x = 0
    if top_y >= row :
        top_y = row-1
    if btm_y < 0 :
        btm_y = 0

    for i in xrange(left_x, right_x+1) :
        im_matrix[top_y, i] = [255,0,0]
        im_matrix[btm_y, i] = [255,0,0]
    for j in xrange(btm_y, top_y+1) :
        im_matrix[j, left_x  ] = [255,0,0]
        im_matrix[j, right_x ] = [255,0,0]

# ----------------------------------------- 
fold_file = open(test_list,  'r')

while 1 :
    foldfile_line = fold_file.readline().rstrip()
    
    cond_eof = foldfile_line == ''
    cond_numface = len(foldfile_line) <= 2
    if cond_numface != 1 :
        cond_newimg = (foldfile_line[0:3] == '200') & (foldfile_line[3]!=' ') & (foldfile_line[3]!='.')
    else :
        cond_newimg = 0

    if cond_eof :       # end of file
        print "1 set done!!"
        break
    
    elif cond_numface : # the number of face in the image
        pass

    elif cond_newimg :  # new image name
        rect_file = open("%s/new-%s.txt"%(text_path, foldfile_line.split('/')[4]), 'r')
        new_image_name = -1

        while 1 :
            temp_line = rect_file.readline().rstrip()
            if temp_line == '' :
                print "eof!"
                new_image = Image.fromarray(im_matrix)
                new_image.save(new_image_name)
                # new_image.show()
                break
            elif len(temp_line) <= 2 :     # the number of face in the image
                print "length!"
                pass
            elif ( temp_line[0:3] == '200' ) & ( temp_line[3]!=' ' ) :   # new image
                print "new image!: ", temp_line
                if new_image_name != -1 :
                    new_image = Image.fromarray(im_matrix)
                    new_image.save(new_image_name)
                    # new_image.show()
    
                old_image = Image.open("%s/%s.jpg" %(base_path, temp_line), 'r')
                col, row = old_image.size
                pixels = old_image.load()
    
                im_matrix = np.zeros([row,col,3]).astype('uint8')
                for j in range(row):
                    for i in range(col):
                        im_matrix[j,i] = pixels[i,j]
    
                img_name = temp_line.split('/')[4]
                new_image_name = "%s/new_%s.jpg" %(new_path, img_name)
    
            else :                        # ellipse information
                temp_line = temp_line.split()
                left_x =  int( temp_line[0] )
                top_y  =  int( temp_line[1] )
                width  =  int( temp_line[2] )
                height =  int( temp_line[3] )
                draw_rectangle(im_matrix, left_x, top_y, width, height)

    print " one image done!!"








#   
#   
#   
#   for i in xrange(1, fold_num+1) :
#       if TEST_MODE :
#           rect_file = open( target_file_name , 'r')
#       else :
#           rect_file = open("%s/FDDB-folds/rect-%s.txt" %(base_path, i), 'r')
#       new_image_name = -1
#   
#       while 1 :
#           temp_line = rect_file.readline().rstrip()
#           if temp_line == '' :
#               new_image = Image.fromarray(im_matrix)
#               new_image.save(new_image_name)
#               new_image.show()
#               break
#           elif len(temp_line) <= 2 :     # the number of face in the image
#               pass
#           elif ( temp_line[0:3] == '200' ) & ( temp_line[3]!=' ' ) :   # new image
#               if new_image_name != -1 :
#                   new_image = Image.fromarray(im_matrix)
#                   # new_image.save(new_image_name)
#                   new_image.show()
#   
#               old_image = Image.open("%s/%s.jpg" %(base_path, temp_line), 'r')
#               col, row = old_image.size
#               pixels = old_image.load()
#   
#               im_matrix = np.zeros([row,col,3]).astype('uint8')
#               for j in range(row):
#                   for i in range(col):
#                       im_matrix[j,i] = pixels[i,j]
#   
#               img_name = temp_line.split('/')[4]
#               if TEST_MODE :
#                   new_image_name = "%s/out_%s.jpg" %(new_path, img_name)
#               else :
#                   new_image_name = "%s/new_%s.jpg" %(new_path, img_name)
#   
#           else :                        # ellipse information
#               temp_line = temp_line.split()
#               left_x =  int( temp_line[0] )
#               top_y  =  int( temp_line[1] )
#               width  =  int( temp_line[2] )
#               height =  int( temp_line[3] )
#               draw_rectangle(im_matrix, left_x, top_y, width, height)
#   
