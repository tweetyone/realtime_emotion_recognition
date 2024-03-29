import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
# from utils import get_labels
from utils import detect_faces
from utils import draw_text
from utils import draw_bounding_box
from utils import apply_offsets
from utils import load_detection_model
from utils import preprocess_input
from read_data import read_name_list,read_file
from train_model import Model


USE_WEBCAM = True # If false, loads video file source

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
people = ['Mingyi', 'Xingyun']
# parameters for loading data and images
emotion_model_path = './model/model_filter.h5'
# emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_cascade = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)
face_model= Model()
face_model.load()

# getting input model shapes for inference
emotion_target_size = (48,48)
face_target_size = (128,128)

# starting lists for calculating modes
emotion_window = []

# starting video streaming

cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

# Select video or webcam feed
cap = None
if (USE_WEBCAM == True):
    cap = cv2.VideoCapture(0) # Webcam source
    # cap = cv2.flip(cap,-1)
else:
    cap = cv2.VideoCapture('./demo/dinner.mp4') # Video file source

while cap.isOpened(): # True:
    ret, bgr_image = cap.read()
    bgr_image = cv2.flip(bgr_image,1)
    #bgr_image = video_capture.read()[1]

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        face_recog = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
            face_recog = cv2.resize(face_recog, (face_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, False)

        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)

        emotion_prediction = emotion_classifier.predict(gray_face)
        # emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction,axis=1)

        emotion_text = class_names[int(emotion_label_arg)]
        picType,prob = face_model.predict(face_recog)
        if picType != -1:
            name_list = read_name_list('/Users/gaoxingyun/Documents/uw/courses/Sp19/EE576_CV/project/realtime_emotion_recognition/dataset')
            print (name_list[picType],prob)
            face_text = name_list[picType]
        else:
            print (" Don't know this person")
            face_text = 'unknown'

        color = (0,255,0)

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_text,
                  color, 0, 45, 1, 1)
        draw_text(face_coordinates, rgb_image, face_text, color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
