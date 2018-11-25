import struct
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import sys

input_path = 'A:/mnist' #mnist数据库解压后的所在路径
output_path = 'A:/mnist/pics' #生成的图片所在的路径


# =====read labels=====
label_file = input_path + '/train-labels.idx1-ubyte'
label_fp = open(label_file, 'rb')
label_buf = label_fp.read()

label_index=0
label_magic, label_numImages = struct.unpack_from('>II', label_buf, label_index)
label_index += struct.calcsize('>II')
labels = struct.unpack_from('>60000B', label_buf, label_index)


# =====read train images=====
label_map = {}
train_file = input_path + '/train-images.idx3-ubyte'
train_fp = open(train_file, 'rb')
buf = train_fp.read()

index=0
magic,numImages,numRows,numColumns=struct.unpack_from('>IIII',buf,index)
index+=struct.calcsize('>IIII')
k = 0
for image in range(0,numImages):
    label = labels[k]
    # if(label_map.has_key(label)):
    if(label in label_map):
        ids = label_map[label] + 1
        label_map[label] += 1
    else:
        label_map[label] = 0
        ids = 0
    k += 1
    if(label_map[label] > 50):
            continue
    im=struct.unpack_from('>784B',buf,index)
    index+=struct.calcsize('>784B')
    im=np.array(im,dtype='uint8')
    im=im.reshape(28,28)
    im=PIL.Image.fromarray(im)
    im.save(output_path + '/%s_%s.bmp'%(label, ids),'bmp')
