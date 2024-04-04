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


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, one_c):
    #https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    
    pts = [((float(one_c[0])), float(one_c[1])),
            ((float(one_c[2])), float(one_c[3])),
            ((float(one_c[4])), float(one_c[5])),
            ((float(one_c[6])), float(one_c[7]))]
    
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(np.array(pts))
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
	    [0, 0],
	    [maxWidth - 1, 0],
	    [maxWidth - 1, maxHeight - 1],
	    [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped



def preprocess_image(img, size=(100, 100)):
    if type(img) == str:
        img = cv.imread(img)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.resize(img_gray, size)
    return img_gray


def extract_LBP_features(imgs, radius=1, n_points=8, method='uniform'):
    features = []
    for img in imgs:
        lbp = local_binary_pattern(img, n_points, radius, method=method)
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(2**8 + 1), range=(0, 2**8))
        features.append(hist)
    return np.array(features)



    


def main():
    lbp_param_list = [
        #{'n_points': 4, 'radius': 1, 'method': 'uniform'},
        #{'n_points': 8, 'radius': 1, 'method': 'uniform'},
        #{'n_points': 12, 'radius': 1, 'method': 'uniform'},
        #{'n_points': 16, 'radius': 1, 'method': 'uniform'},
        #{'n_points': 4, 'radius': 2, 'method': 'uniform'},
        #{'n_points': 8, 'radius': 2, 'method': 'uniform'},
        #{'n_points': 12, 'radius': 2, 'method': 'uniform'},
        {'n_points': 16, 'radius': 2, 'method': 'uniform'},
        #{'n_points': 4, 'radius': 1, 'method': 'default'},
        #{'n_points': 8, 'radius': 1, 'method': 'default'},
        #{'n_points': 12, 'radius': 1, 'method': 'default'},
        #{'n_points': 16, 'radius': 1, 'method': 'default'},
        #{'n_points': 4, 'radius': 2, 'method': 'default'},
        #{'n_points': 8, 'radius': 2, 'method': 'default'},
        #{'n_points': 12, 'radius': 2, 'method': 'default'},
        #{'n_points': 16, 'radius': 2, 'method': 'default'},

    ]
    best_accuracy = 0
    best_params = None
    for params in lbp_param_list:
        free_images = [preprocess_image(img) for img in glob.glob("free/*.png")]  
        full_images = [preprocess_image(img) for img in glob.glob("full/*.png")] 
        
        free_features = extract_LBP_features(free_images, **params)
        full_features = extract_LBP_features(full_images, **params)

        labels = np.concatenate((np.zeros(len(free_features)), np.ones(len(full_features))))

        data = np.concatenate((free_features, full_features))

        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

        start_time = time.time()
        knn = KNeighborsClassifier(n_neighbors=3, algorithm='auto', leaf_size=10, p=1, weights='uniform')
        knn.fit(X_train, y_train)
        end_time = time.time()



        pkm_file = open('parking_map_python.txt', 'r')
        pkm_lines = pkm_file.readlines()
        pkm_coords = []

        for line in pkm_lines:
            st_line = line.strip()
            sp_line = list(st_line.split(" "))
            pkm_coords.append(sp_line)

        test_images = [img for img in glob.glob("test_images_zao/*.jpg")]
        test_results = [img for img in glob.glob("test_images_zao/*.txt")]

        test_images.sort()
        test_results.sort()

        size = (100, 100)
        total_parkings = 0
        total_correct_parkings = 0
        n_park = 0
        font = cv.FONT_HERSHEY_PLAIN
        testing_images = []
        testing_images_coords = []
        res_lines = []
        for img_name, result in zip(test_images, test_results):

            img = cv.imread(img_name)
            res = open(result, 'r')
            new_lines = res.readlines()

            
            new_lines = [int(line.strip()) for line in new_lines]
            res_lines.extend(new_lines)
            img_clone = img.copy()

            for coord in pkm_coords:

                testing_images.append(preprocess_image(four_point_transform(img, coord)))
                testing_images_coords.append(coord)

        
        
        test_features = extract_LBP_features(testing_images, **params)
        predictions = knn.predict(test_features)
        for i, pred in enumerate(predictions):
            if pred == res_lines[i]:
                total_correct_parkings += 1
            total_parkings += 1
        n_park = 0
        testing_images = []
        testing_images_coords = []
        print("Total parkings: ", total_parkings)
        print("Total correct parkings: ", total_correct_parkings)
        print("Accuracy: ", total_correct_parkings / total_parkings)
        if total_correct_parkings / total_parkings > best_accuracy:
            best_accuracy = total_correct_parkings / total_parkings
            best_params = params
    print("Best LBP parameters:", best_params)
    print("Best accuracy:", best_accuracy)





if __name__ == "__main__":
   main()