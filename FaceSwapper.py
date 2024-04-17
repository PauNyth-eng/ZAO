import cv2 as cv
import numpy as np
import time
import dlib


def remove_duplicates(locations, threshold_distance):
    non_duplicates = []
    for i in range(len(locations)):
        is_duplicate = False
        for j in range(i+1, len(locations)):
            dist = ((locations[i][0] + locations[i][2] / 2) - (locations[j][0] + locations[j][2] / 2))**2 + \
                   ((locations[i][1] + locations[i][3] / 2) - (locations[j][1] + locations[j][3] / 2))**2
            dist = dist ** 0.5
            if dist < threshold_distance:
                is_duplicate = True
                break
        if not is_duplicate:
            non_duplicates.append(locations[i])
    return non_duplicates


def draw_rectangle (paint_frame, locations, color1, color2):
    for (x, y, w, h) in locations:
        x += 50
        y += 50
        w -= 100
        h -= 100
        cv.rectangle(paint_frame, (x, y), (x+w, y+h), color1, 8)
        cv.rectangle(paint_frame, (x, y), (x+w, y+h), color2, 2)

def detector():
    video = cv.VideoCapture("fusek_face_car_01.avi")
    face_cascade = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    face_cascade_profile = cv.CascadeClassifier("haarcascades/haarcascade_profileface.xml")
    face_anime = cv.CascadeClassifier('lbpcascade_animeface.xml')
    predictor = dlib.shape_predictor("dlib_shape_predictor_68_face_landmarks.dat")
    imagurator = cv.imread('funnyPictures/NormalFace.jpg')
    imagurator_copy = imagurator.copy()
    face_roi_src = None
    imagurator_x = 0
    imagurator_y = 0
    imagurator_width = 0
    imagurator_height = 0


    if imagurator is not None:
        location_imagurator = face_cascade.detectMultiScale(imagurator, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100), maxSize=(500, 500))
        print(location_imagurator)
        draw_rectangle(imagurator, location_imagurator, (0, 0, 255), (255, 0, 0))
        for i, (x, y, w, h) in enumerate(location_imagurator):
            x += 50
            y += 50
            w -= 80
            h -= 50
            imagurator_x = x
            imagurator_y = y
            imagurator_width = w
            imagurator_height = h
            face_roi_src = imagurator_copy[y:y + h, x:x + w]
        cv.imshow("src_img", imagurator)
    threshold_distance = 100.0
    while True:
        start = time.time()
        ret, frame = video.read()

        if frame is None:
            break
        locations_faces = []
        if ret is True:
            location_faces = face_cascade.detectMultiScale(frame, 1.2, 5, minSize=(100, 100), maxSize=(500, 500))
            for l_face in location_faces:
                locations_faces.append(l_face)
            location_face_profile = face_cascade_profile.detectMultiScale(frame, 1.2, 5, minSize=(100, 100), maxSize=(500, 500))
            for l_face in location_face_profile:
                locations_faces.append(l_face)



            for i, (x, y, w, h) in enumerate(locations_faces):
                x += 50
                y += 70
                w -= 100
                h -= 50

                face_roi = frame[y:y + h, x:x + w]

                height, width, _ = face_roi.shape

                face_roi_copy = face_roi.copy()

                face = dlib.rectangle(x, y, x + w, y + h)
                shape = predictor(frame, face)
                for ip, p in enumerate(shape.parts()):
                    cv.circle(face_roi, (p.x, p.y), 2, (0, 0, 255), thickness=-1)
                cv.imshow('face', face_roi_copy)
                face_roi_src_r = cv.resize(face_roi_src, (width, height))
                face_roi_src_r_copy = face_roi_src_r.copy()
                cv.imshow('face_roi', face_roi_src_r_copy)

                src_mask = predictor(imagurator_copy, dlib.rectangle(imagurator_x, imagurator_y, imagurator_x + imagurator_width, imagurator_y + imagurator_height))
                points = np.array([(p.x, p.y) for p in src_mask.parts()], np.int32)
                hull = cv.convexHull(points)
                mask = np.zeros(imagurator_copy.shape[:2], np.uint8)
                src_mask = cv.fillConvexPoly(mask, hull, 255)
                src_crop = src_mask[imagurator_y:imagurator_y+imagurator_height, imagurator_x:imagurator_x+imagurator_width]
                src_mask_resized = cv.resize(src_crop, (width, height))
                cv.imshow('mask', src_mask_resized)

                seamlessclone = cv.seamlessClone(face_roi_src_r_copy, face_roi_copy, src_mask_resized, (width//2, height//2), cv.MONOCHROME_TRANSFER)
                cv.imshow('seamlessclone', seamlessclone)

                if y + height <= frame.shape[0] and x + width <= frame.shape[1]:
                    frame[y:y + height, x:x + width] = seamlessclone

            end = time.time()

            final = end - start

            cv.putText(frame, f"FPS: {1/final:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

            cv.imshow('frame', frame)
            if cv.waitKey(2) == ord('q'):
                break






if __name__ == "__main__":
    detector()
    cv.destroyAllWindows()