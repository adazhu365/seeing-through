import cv2
import numpy as np
from statistics import mode
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Functional Dependencies
def load_image(image_path, grayscale=False, target_size=None):
    pil_image = image.load_img(image_path, grayscale, target_size)
    return image.img_to_array(pil_image)

def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)

def get_colors(num_classes):
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    colors = np.asarray(colors) * 255
    return colors

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}
    elif dataset_name == 'imdb':
        return {0: 'woman', 1: 'man'}
    elif dataset_name == 'age':
        return {0: '10',1: '11',2: '12',3: '13',4: '14',5: '15',6: '16',7: '17',8: '18',9: '19',10: '20',
        11: '21',12: '22',13: '23',14: '24',15: '25',16: '26',17: '27',18: '28',19: '29',20: '30',21: '31',
        22: '32',23: '33',24: '34',25: '35',26: '36',27: '37',28: '38',29: '39',30: '40',31: '41',32: '42',
        33: '43',34: '44',35: '45',36: '46',37: '47',38: '48',39: '49',40: '50',41: '51',42: '52',43: '53',
        44: '54',45: '55',46: '56',47: '57',48: '58',49: '59',50: '60',51: '61',52: '62',53: '63',54: '64',
        55: '65',56: '66',57: '67',58: '68',59: '69',60: '70',61: '71',62: '72',63: '73',64: '74',65: '75',
        66: '76',67: '77',68: '78',69: '79',70: '80',71: '81',72: '82',73: '83',74: '84',75: '85',76: '86',
        77: '87',78: '88',79: '89',80: '90',81: '91',82: '92',83: '93',84: '94',85: '95',86: '96',87: '97',
        88: '98',89: '99',90: '100',91: '101'}

# Dashboard (Terminal Output) Helpers
dash = '-' * 40





# parameters for loading data and images
detection_model_path = 'detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = 'plus_emotion_mini_XCEPTION.58-0.80.hdf5'
gender_model_path = 'resave_simple_CNN.81-0.96.hdf5'
age_model_path = 'age_mini_XCEPTION.05-0.05.hdf5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
age_labels = get_labels('age')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
frame_window = 10
gender_offsets = (30, 60)
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)
age_classifier = load_model(age_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]
age_target_size = age_classifier.input_shape[1:3]

# starting lists for calculating modes
gender_window = []
emotion_window = []
age_window = []

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
while True:

    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
        rgb_face = rgb_image[y1:y2, x1:x2]

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            rgb_face = cv2.resize(rgb_face, (gender_target_size))
            rgb_face2 = cv2.resize(rgb_face, (age_target_size))
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue
        gray_face = preprocess_input(gray_face, False)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_preds = emotion_classifier.predict(gray_face)
        
        emotion_label_arg = np.argsort(-emotion_preds)[:,:2][0]

        print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
        pct1=int(round(emotion_preds[0,emotion_label_arg[0]],2)*100)
        pct2=int(round(emotion_preds[0,emotion_label_arg[1]],2)*100)
        print(dash)
        print('#   Emotion        Certainty')
        print(dash)
        print('1   {:<15s}{}'.format(emotion_labels[emotion_label_arg[0]], round(emotion_preds[0,emotion_label_arg[0]],2)))
        print('2   {:<15s}{}'.format(emotion_labels[emotion_label_arg[1]], round(emotion_preds[0,emotion_label_arg[1]],2)))
        print(dash)
        # print('pct1', round(emotion_preds[0,emotion_label_arg[0]],2))
        # print('pct2', round(emotion_preds[0,emotion_label_arg[1]],2))
        emotion_text = '%s' %(emotion_labels[emotion_label_arg[0]])
        emotion_window.append(emotion_text)

        rgb_face = np.expand_dims(rgb_face, 0)
        rgb_face = preprocess_input(rgb_face, False)
        gender_prediction = gender_classifier.predict(rgb_face)
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = gender_labels[gender_label_arg]
        gender_window.append(gender_text)

        
        rgb_face2 = np.expand_dims(rgb_face2, 0)
        rgb_face2 = preprocess_input(rgb_face2, False)
        age_prediction = age_classifier.predict(rgb_face2)
        age_label_arg = np.argmax(age_prediction)
        age_text = age_labels[age_label_arg]
        age_window.append(age_text)


        if len(gender_window) > frame_window:
            emotion_window.pop(0)
            gender_window.pop(0)
            age_window.pop(0)
            
        try:
            emotion_mode = mode(emotion_window)
            gender_mode = mode(gender_window)
            age_mode = mode(age_window)
        except:
            continue

        if gender_text == gender_labels[0]:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, gender_mode,
                  color, 0, -20, 1, 1)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)
        draw_text(face_coordinates, rgb_image, age_mode,
                  color, 0, -70, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
