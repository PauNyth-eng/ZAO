import glob
import cv2 as cv
import numpy as np
import time
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import local_binary_pattern
from sklearn.metrics import accuracy_score


def preprocess_image(img, size=(100, 100)):
    img = cv.imread(img)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.resize(img_gray, size)
    return img_gray


def extract_LBP_features(imgs, radius=1, n_points=8, method='uniform'):
    features = []
    for img in imgs:
        lbp = local_binary_pattern(img, n_points, radius, method=method)
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        features.append(hist)
    return np.array(features)


def main():
    free_images = [preprocess_image(img) for img in glob.glob("free/*.png")]  
    full_images = [preprocess_image(img) for img in glob.glob("full/*.png")] 
    
    free_features = extract_LBP_features(free_images)
    full_features = extract_LBP_features(full_images)

    labels = np.concatenate((np.zeros(len(free_features)), np.ones(len(full_features))))

    data = np.concatenate((free_features, full_features))

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier()
    
    # Define a parameter grid for grid search
    lbp_param_list = [
        # Varying n_points and radius for detailed vs. robust features
        {'n_points': 4, 'radius': 1, 'method': 'uniform'},
        {'n_points': 8, 'radius': 1, 'method': 'uniform'},
        {'n_points': 12, 'radius': 1, 'method': 'uniform'},
        {'n_points': 16, 'radius': 1, 'method': 'uniform'},
        {'n_points': 4, 'radius': 2, 'method': 'uniform'},
        {'n_points': 8, 'radius': 2, 'method': 'uniform'},
        {'n_points': 12, 'radius': 2, 'method': 'uniform'},
        {'n_points': 16, 'radius': 2, 'method': 'uniform'},

        # Exploring different methods for diverse texture characteristics
        {'n_points': 8, 'radius': 1, 'method': 'nri_uniform'},
        {'n_points': 12, 'radius': 1, 'method': 'nri_uniform'},
        {'n_points': 8, 'radius': 1, 'method': 'var'},
        {'n_points': 12, 'radius': 1, 'method': 'var'},
        {'n_points': 8, 'radius': 1, 'method': 'eft'},
        {'n_points': 12, 'radius': 1, 'method': 'eft'},
        {'n_points': 8, 'radius': 1, 'method': 'nriu2'},
        {'n_points': 12, 'radius': 1, 'method': 'nriu2'},

        # Combining n_points, radius, and methods for tailored exploration
        {'n_points': 16, 'radius': 2, 'method': 'var'},
        {'n_points': 20, 'radius': 1, 'method': 'eft'},
        {'n_points': 12, 'radius': 3, 'method': 'nri_uniform'},
        {'n_points': 20, 'radius': 2, 'method': 'nriu2'},
    ]

    best_accuracy = 0
    best_lbp_params = None

    for params in lbp_param_list:
        # Extract features with current LBP parameters
        free_features = extract_LBP_features(free_images, **params)
        full_features = extract_LBP_features(full_images, **params)
        features = np.concatenate((free_features, full_features))

        # Train and evaluate KNN
        knn = KNeighborsClassifier(n_neighbors=3)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=1, random_state=42)
        knn.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, knn.predict(X_test))

        # Update best parameters if current accuracy is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_lbp_params = params

    print("Best LBP parameters:", best_lbp_params)
    print("Best accuracy:", best_accuracy)


if __name__ == "__main__":
    main()
