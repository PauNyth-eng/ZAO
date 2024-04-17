import glob
import cv2 as cv
import numpy as np
import time
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score




def preprocess_image(img, size=(100, 100)):
    if type(img) == str:
        img = cv.imread(img)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.resize(img_gray, size)
    return img_gray

def Labeling(img):
    word_label = img.split('_')[4]
    if word_label[0] == '1': return 0
    elif word_label[0] == '0': return 1

def main():
    test_dir = [img for img in glob.glob("Eye_dataset/Test_Data/*.png")] 
    train_dir = [img for img in glob.glob("Eye_dataset/Train_Data/*.png")]
    closed = []
    opened = []
    for img in test_dir:
        if Labeling(img) == 1:
            opened.append(preprocess_image(img))
        else:
            closed.append(preprocess_image(img))

    print("Opened: ", len(opened))
    print("Closed: ", len(closed))
    
     
    
    recognizer = cv.face.LBPHFaceRecognizer_create(radius=2, neighbors=8)
    data = np.concatenate((opened, closed))
    labels = np.concatenate((np.zeros(len(opened), dtype=np.int32), np.ones(len(closed), dtype=np.int32)))
    start_time = time.time()
    recognizer.train(data, labels)
    print("Training time: ", time.time() - start_time)
    recognizer.save(f"eye_recognizer_{2}_{8}.yml")


    
        

    
    
    return 0



if __name__ == '__main__':
    main()
