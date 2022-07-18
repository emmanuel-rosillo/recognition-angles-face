import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
import os
from cvzone.FaceMeshModule import FaceMeshDetector
from ConfCam import CamConfig, findFocalLength, getPathConfigCam
import json

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# object cam configure
findDistances = CamConfig()

# distance between user and WebCam

_distanceRequired = 37


def getPic(fileName):
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(maxFaces=1)

    with open(fileName) as file:
        focal = json.load(file)

    # capture images in cvzone
    while True:
        success, frame = cap.read()
        height, width, _ = frame.shape
        frame, faces = detector.findFaceMesh(frame, draw=False)
        frame = cv2.flip(frame, 1)

        if faces:
            face = faces[0]
            # *-*-*-*-*-* eyes detection *-*-*-*-*-*
            eyeRight = face[145]
            eyeLeft = face[374]

            # draws

            # *--* w *--*
            w, _ = detector.findDistance(eyeRight, eyeLeft)
            f = focal['focalLength']
            W = findDistances.distanceBetweenEyes
            d = (W * f) / w

            cv2.putText(frame, str(int(d)), (20, 20), 1, 1, (0, 255, 0), 2)

        with mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=0.5) as face_mesh:

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks is not None:
                for face_landmarks in results.multi_face_landmarks:

                    # front
                    frontX = int(face_landmarks.landmark[10].x * width)
                    frontY = int(face_landmarks.landmark[10].y * height)

                    # chin
                    chinX = int(face_landmarks.landmark[152].x * width)
                    chinY = int(face_landmarks.landmark[152].y * height)

                    # earLeft
                    earLeftX = int(face_landmarks.landmark[234].x * width)
                    earLeftY = int(face_landmarks.landmark[234].y * height)

                    # earRight
                    earRightX = int(face_landmarks.landmark[454].x * width)
                    earRightY = int(face_landmarks.landmark[454].y * height)

                    # nose
                    noseX = int(face_landmarks.landmark[4].x * width)
                    noseY = int(face_landmarks.landmark[4].y * height)

                    # points
                    front = np.array([frontX, frontY])
                    chin = np.array([chinX, chinY])
                    earLeft = np.array([earLeftX, earLeftY])
                    earRight = np.array([earRightX, earRightY])
                    center = np.array([frontX, earRightY])
                    nose = np.array([noseX, noseY])

                    # lines
                    noseCenter = np.linalg.norm(center - nose)

                    if noseCenter != 0:
                        # *-*-*-*-* draws *-*-*-*-*

                        # *-*-* front *-*-*
                        cv2.circle(frame, (frontX, frontY), 2, (255, 0, 255), 2)
                        cv2.circle(frame, (chinX, chinY), 2, (255, 0, 255), 2)
                        # *-*-*-*-*-*

                        # *-*-* ears *-*-*
                        cv2.circle(frame, (earLeftX, earLeftY), 2, (255, 0, 255), 2)
                        cv2.circle(frame, (earRightX, earRightY), 2, (255, 0, 255), 2)
                        # *-*-*-*-*-*

                        # *-*-* nose *-*-*
                        cv2.circle(frame, (noseX, noseY), 2, (255, 0, 255), 2)
                        cv2.putText(frame, str(noseCenter), (noseX, noseY), 1, 1, (0, 255, 0), 2)
                        # *-*-*-*-*-*

                        # *-*-*-* lines draws *-*-*-*
                        cv2.line(frame, (earLeftX, earLeftY), (earRightX, earRightY), (211, 0, 148), 2)
                        cv2.line(frame, (frontX, frontY), (chinX, chinY), (211, 0, 148), 2)

                        # face center draw
                        cv2.circle(frame, (frontX, earRightY), 2, (0, 0, 255), 3)


        cv2.imshow("facial web", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            cap.release()
            cv2.destroyAllWindows()
            break


def alignedFace(frame):
    with mp_face_detection.FaceDetection(
            min_detection_confidence=0.5) as face_detection:
        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        if results.detections is not None:
            for detection in results.detections:
                # Ojo 1
                x1 = int(detection.location_data.relative_keypoints[0].x * width)
                y1 = int(detection.location_data.relative_keypoints[0].y * height)
                # Ojo 2
                x2 = int(detection.location_data.relative_keypoints[1].x * width)
                y2 = int(detection.location_data.relative_keypoints[1].y * height)

                p1 = np.array([x1, y1])
                p2 = np.array([x2, y2])
                p3 = np.array([x2, y1])

                d_eyes = np.linalg.norm(p1 - p2)
                l1 = np.linalg.norm(p1 - p3)

                angle = degrees(acos(l1 / d_eyes))

                if y1 < y2:
                    angle = - angle

                M = cv2.getRotationMatrix2D((width // 2, height // 2), -angle, 1)
                aligned_image = cv2.warpAffine(frame, M, (width, height))

                results2 = face_detection.process(cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB))
                if results2.detections is not None:
                    for detection in results2.detections:
                        xmin = int(detection.location_data.relative_bounding_box.xmin * width)
                        ymin = int(detection.location_data.relative_bounding_box.ymin * height)
                        w = int(detection.location_data.relative_bounding_box.width * width)
                        h = int(detection.location_data.relative_bounding_box.height * height)
                        if xmin < 0 or ymin < 0:
                            continue
                        aligned_face = aligned_image[ymin: ymin + h, xmin: xmin + w]
                        cv2.imwrite("aligned_face", aligned_face)


def control():
    # get path json focal length
    fileName = getPathConfigCam()

    # load focal configured
    if not os.path.isfile(fileName):
        findFocalLength()

    # get pic in parameters
    getPic(fileName)


if __name__ == '__main__':
    control()
