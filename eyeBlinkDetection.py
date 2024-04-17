import dlib
import cv2 as cv
import time
import glob
import numpy as np
import math

EAR_STASH = 0.26


def calculate_EAR(points):
    def euclidean_distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    points = [(p.x, p.y) for p in points]
    p1, p2, p3, p4, p5, p6 = points
    return euclidean_distance((p2, p6), (p3, p5))/(2*euclidean_distance(p1, p4))


def ear_state(ear):
    print(ear)
    if ear < EAR_STASH:
        return 0
    else:
        return 1


def calculate_color_ear(eye_states):
    length = len(eye_states)
    summary = sum(eye_states)
    if summary < length / 2:
        return (0, 255, 255)
    else:
        return (0, 255, 0)


def face_detect():
    frames = [img for img in glob.glob("img/*.jpg")]
    frames.sort()
    cv.namedWindow("face_detect", cv.WINDOW_NORMAL)
    detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor("dlib_shape_predictor_68_face_landmarks.dat")
    face_cascade = cv.CascadeClassifier("lbpcascade_frontalface_improved.xml")
    left_eye_states = []
    right_eye_states = []

    max_eye_states_l = 3
    max_eye_states_r = 3

    for frame in frames:
        img = cv.imread(frame)
        start_time = time.time()
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=7, minSize=(100, 100),
                                          maxSize=(500, 500))

        for i, face in enumerate(faces):
            face_x, face_y, face_w, face_h = face
            face = dlib.rectangle(face_x, face_y, face_x + face_w, face_y + face_h)
            shape = landmark_predictor(img, face)

            landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            if max_eye_states_r != 0:
                right_eye_states.append(ear_state(calculate_EAR(shape.parts()[42:48])))
                max_eye_states_r -= 1
            else:
                right_eye_states.pop(0)
                right_eye_states.append(ear_state(calculate_EAR(shape.parts()[42:48])))

            if max_eye_states_l != 0:
                left_eye_states.append(ear_state(calculate_EAR(shape.parts()[36:42])))
                max_eye_states_l -= 1
            else:
                left_eye_states.pop(0)
                left_eye_states.append(ear_state(calculate_EAR(shape.parts()[36:42])))

            left_eye_x, left_eye_y, left_eye_w, left_eye_h = cv.boundingRect(np.array([left_eye]))
            right_eye_x, right_eye_y, right_eye_w, right_eye_h = cv.boundingRect(np.array([right_eye]))
            left_eye_x -= 10
            left_eye_y -= 10
            left_eye_w += 20
            left_eye_h += 20
            right_eye_x -= 10
            right_eye_y -= 10
            right_eye_w += 20
            right_eye_h += 20
            cv.rectangle(img, (left_eye_x, left_eye_y), (left_eye_x + left_eye_w, left_eye_y + left_eye_h), calculate_color_ear(left_eye_states),
                         2)
            cv.rectangle(img, (right_eye_x, right_eye_y), (right_eye_x + right_eye_w, right_eye_y + right_eye_h),
                         calculate_color_ear(right_eye_states), 2)
            for ip, p in enumerate(shape.parts()[36:42]):
                cv.circle(img, (p.x, p.y), 2, calculate_color_ear(left_eye_states), -1)

            for ip, p in enumerate(shape.parts()[42:48]):
                cv.circle(img, (p.x, p.y), 2, calculate_color_ear(right_eye_states), -1)

            print(f"LEFT EAR: {left_eye_states}, RIGHT EAR: {right_eye_states}")
        end_time = time.time()
        final_time = end_time - start_time
        cv.putText(img, f"FPS: {1 / final_time:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv.LINE_AA)
        cv.imshow("face_detect", img)

        if cv.waitKey(2) == ord('q'):
            break


if __name__ == "__main__":
    face_detect()