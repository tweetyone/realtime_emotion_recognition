#-*-coding:utf8-*-

from read_data import read_name_list,read_file
from train_model import Model
from keras.models import load_model
import cv2


def test_onePicture(path):
    model= Model()
    model.load()
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    picType,prob = model.predict(img)
    if picType != -1:
        name_list = read_name_list('/Users/gaoxingyun/Documents/uw/courses/Sp19/EE576_CV/project/realtime_emotion_recognition/dataset')
        print (name_list[picType],prob)
    else:
        print (" Don't know this person")

def test_onBatch(path):
    model= Model()
    model.load()
    index = 0
    img_list, label_lsit, counter = read_file(path)
    for img in img_list:
        picType,prob = model.predict(img)
        if picType != -1:
            index += 1
            name_list = read_name_list('/Users/gaoxingyun/Documents/uw/courses/Sp19/EE576_CV/project/faceRecognition/dataset')
            print (name_list[picType])
        else:
            print (" Don't know this person")

    return index

if __name__ == '__main__':
    # face_model_path = './model/face_model.h5'
    # face_classifier = load_model(face_model_path)
    # img = cv2.imread('/Users/gaoxingyun/Documents/uw/courses/Sp19/EE576_CV/project/Data/dataset/Mingyi/Mingyi31.png')
    # img = cv2.resize(img, (64, 64))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # picType,prob = face_classifier.predict(img)
    # if picType != -1:
    #     name_list = read_name_list('/Users/gaoxingyun/Documents/uw/courses/Sp19/EE576_CV/project/faceRecognition/dataset')
    #     print (name_list[picType],prob)
    # else:
    #     print (" Don't know this person")

    test_onePicture('/Users/gaoxingyun/Documents/uw/courses/Sp19/EE576_CV/project/Data/dataset/Xingyun/Xingyun6.png')
