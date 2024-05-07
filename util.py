import os
import datetime
import torch
import tensorflow as tf
import csv
import cv2
import numpy as np
from scipy.special import softmax
from keras.applications.resnet50 import preprocess_input

def get_images(detector, mode, path):
    if mode == '0':
        mode = int(mode)
    cap = cv2.VideoCapture(mode)
    count = 0
    while count < 1:
        ret, img = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv2.imshow(f'Look straight into the camera and press s', img)
        if cv2.waitKey(1) == ord('s'):
            detection = detector(img)
            result = detection.pandas().xyxy[0].to_numpy()
            if len(result) >= 1:
                i = result[0]
                x_min = int(i[0])
                x_max = int(i[2])
                y_min = int(i[1])
                y_max = int(i[3])
                img = img[y_min:y_max, x_min:x_max]
                img = cv2.resize(img, (125, 150))
                cv2.imwrite(f'{path}/{count}.jpg', img)
                count+=1

    cap.release()
    cv2.destroyAllWindows()

def get_feature(feature_extractor, path):
    list_images = os.listdir(path)
    images = []
    for i in list_images:
        img = cv2.imread(f'{path}/{i}')
        images.append(img)
    images = np.array(images)
    images = preprocess_input(images)
    features = feature_extractor(images)
    feature = np.sum(features, axis = 0)
    feature = np.reshape(feature, (1, 128))
    np.set_printoptions(precision=4,formatter={'float_kind':'{:4f}'.format})
    return feature


def add_new_person(name, detector, feature_extractor, mode = 0 ):
    path = f'Person/images/{name}'
    
    feature = get_feature(feature_extractor, path)
    
    write_feature(feature)
    write_name(name)

def euclidean_distance(vector1, vector2):
    return np.sqrt(np.sum((vector1 - vector2) ** 2))

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector  # Avoid division by zero
    return vector / norm

def predict(feature, all_feature, label):
    '''
    This function takes in three parameters: feature, all_feature, and label. 
    
    # feature is a feature vector of a single data point
    # all_feature is a matrix of all the features for all data points
    # label is an array of labels for each data point

    # The function calculates the Euclidean distance between the feature vector and all other feature vectors in the matrix
    It then uses the softmax function to calculate probabilities for each distance, and finds the index of the highest probability
    It then rounds this probability to two decimal places and returns it as 'acc'
    The function returns a tuple containing 'name' and 'acc'.
    '''
    # Normalize the feature vector
    normalized_feature_vector = normalize_vector(feature)
    
    # Normalize each class feature vector
    normalized_class_feature_vectors = [normalize_vector(class_vector) for class_vector in all_feature]
    
    # Compute distances
    distances = [euclidean_distance(normalized_feature_vector, class_vector) for class_vector in normalized_class_feature_vectors]
    prob = 1 - softmax(distances)
    index = np.argmax(prob)

    acc = acc = np.around(np.max(prob), decimals = 2)
    name = label[index]
    return name, acc
    
def get_label(file):
    '''
    This function takes a file as an argument and returns a list of labels
    It opens the file in read mode, reads the lines from the file
    removes the newline character from each line and stores it in a list
    Finally, it closes the file and returns the list of labels.
    '''
    f = open(file, 'r')
    label = f.readlines()
    label = [name[:-1] for name in label]
    f.close()
    return label

def get_feature_array(file):
    '''
    This function takes a file as an argument and returns an array of the data from the file
    '''
    array = np.loadtxt(file, delimiter =',')
    return array

def write_name(name):
    '''
    This function write new person's name to file
    '''
    f = open('./Person/name.txt', 'a')
    f.write(f'{name}')
    f.write('\n')
    f.close()

def write_feature(feature):
    '''
    This function write new person's feature to file
    '''
    f = open('./Person/feature.csv', 'a')
    writer = csv.writer(f)
    writer.writerow(*feature)
    f.close()

def get_path():
    '''
    This function gets the path to the Attendance folder for the current date
    It checks if a folder for the current date exists in the Attendance folder
    and if not, it creates a folder with that name, creates an Attended_images folder inside it
    and creates a csv file with a header row called attended_table.csv
    Finally, it returns the path to the new or existing folder.
    '''
    list_atd = os.listdir('./Attendance')
    today = datetime.datetime.now()
    today = today.strftime('%d%m%y')
    if today in list_atd:
        pass
    else:
        header = ['Number', 'Name', 'Time']
        os.mkdir(f'./Attendance/{today}')
        os.chdir(f'./Attendance/{today}')
        os.mkdir('Attended_images')
        f = open('attended_table.csv', 'w')
        writer = csv.writer(f)
        writer.writerow(header)
        f.close()
    path = f'./Attendance/{today}'
    return path

def create_attended_count(label):
    '''
    This function create dictionary with key = name in label and value = 0
    '''
    atd_count = {}
    for name in label:
        atd_count[name] = 0 
    return atd_count

def update(file, name, number):
    '''
    This function write infomation after attended success
    '''
    f = open(file, 'a')
    writer = csv.writer(f)
    time = datetime.datetime.now()
    time = time.strftime('%H:%M')
    writer.writerow([number, name, time])
    f.close()

def get_number(file):
    '''
    This function return number in attended file 
    '''
    f = open(file, 'r')
    reader = csv.reader(f)
    number = 0
    for line in reader:
        number +=1
    f.close()
    return number

def show_atd(path):
    f = open(path, 'r')
    reader = csv.reader(f)
    for row in reader:
        print(*row)

def recognition(detector, feature_extractor):
    
    cap = cv2.VideoCapture('/home/noah/dev/Yolov8Pipeline/Datasets/Datacam Datasets/Employee Datasets/1799_videos/20240119/Camera_3.avi')
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    path = get_path()
    label = get_label('Person/name.txt')
    attended_count = create_attended_count(label)

    while True:
        ret, image = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        detections = detector.predict(image, imgsz=640, conf=0.72, iou=0.4)
        for detection in detections:
            if(len(detection.boxes) > 0):
                for det in range(len(detection.boxes)):
                    if (detection.boxes.cls.cpu().numpy()[det] == 0):
                        x_min = int(detection.boxes.xyxy.data[det].cpu().numpy()[0])
                        x_max = int(detection.boxes.xyxy.data[det].cpu().numpy()[2])
                        y_min = int(detection.boxes.xyxy.data[det].cpu().numpy()[1])
                        y_max = int(detection.boxes.xyxy.data[det].cpu().numpy()[3])
                        img = image[y_min:y_max, x_min:x_max]
                        img = cv2.resize(img, (128, 128))
                        img_exp = np.expand_dims(img, axis = 0)
                        feature = feature_extractor(img_exp)

                        label = get_label('Person/name.txt')
                        feature_array = get_feature_array('Person/feature.csv')
                        name, acc = predict(feature, feature_array, label)
                        label = name
                        # process
                        if acc > 0.70:
                            label = str(name)
                            image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                            image = cv2.putText(image, f'{label}', (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
                        else:
                            label = 'Unknown'
                            image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                            image = cv2.putText(image, f'{label}', (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
                        

                        if attended_count[name] == 20:
                            number = get_number(f'{path}/attended_table.csv')
                            update(f'{path}/attended_table.csv', name, number)
                            cv2.imwrite(f'{path}/Attended_images/{name}.jpg',img)
                        
                    else:
                        continue
        cv2.imshow('Recognition - Enter q to exit', image)
            
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()