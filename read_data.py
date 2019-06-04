#-*-coding:utf8-*-

import os
import cv2
import numpy as np

from read_img import endwith

def read_file(path):
    img_list = []
    label_list = []
    dir_counter = 0
    IMG_SIZE = 128

    dir = os.listdir(path)
    dir.pop(0)
    for child_dir in dir:
         child_path = os.path.join(path, child_dir)

         for dir_image in  os.listdir(child_path):
             if endwith(dir_image,'png'):
                img = cv2.imread(os.path.join(child_path, dir_image))
                resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                recolored_img = cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
                img_list.append(recolored_img)
                label_list.append(dir_counter)

         dir_counter += 1

    img_list = np.array(img_list)

    return img_list,label_list,dir_counter

def read_name_list(path):
    name_list = []
    dir = os.listdir(path)
    dir.pop(0)
    for child_dir in dir:
        name_list.append(child_dir)
    name_list.reverse()
    return name_list



if __name__ == '__main__':
    img_list,label_list,counter = read_file('/Users/gaoxingyun/Documents/uw/courses/Sp19/EE576_CV/project/realtime_emotion_recognition/dataset')
    name_list = read_name_list('/Users/gaoxingyun/Documents/uw/courses/Sp19/EE576_CV/project/realtime_emotion_recognition/dataset')
    print (counter)
    print(name_list)
    # print (img_list)
    # print (label_list)
