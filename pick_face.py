#-*-coding:utf8-*-
import os
import cv2
import time
from read_img import readAllImg


def readPicSaveFace(sourcePath,objectPath,*suffix):
    try:
        resultArray=readAllImg(sourcePath,*suffix)
        count = 1
        face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        for i in resultArray:
            if type(i) != str:

              gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
              faces = face_cascade.detectMultiScale(gray, 1.3, 5)
              for (x, y, w, h) in faces:

                listStr = [str(int(time.time())), str(count)]  #以时间戳和读取的排序作为文件名称
                fileName = ''.join(listStr)

                f = cv2.resize(gray[y:(y + h), x:(x + w)], (200, 200))
                cv2.imwrite(objectPath+os.sep+'%s.png' % fileName, f)
                count += 1


    except IOError:
        print ("Error")

    else:
        print ('Already read '+str(count-1)+' Faces to Destination '+objectPath)

if __name__ == '__main__':
     readPicSaveFace('/Users/gaoxingyun/Documents/uw/courses/Sp19/EE576_CV/project/Data/dataset/allPics','/Users/gaoxingyun/Documents/uw/courses/Sp19/EE576_CV/project/Data/dataset/allFaces','.png')
