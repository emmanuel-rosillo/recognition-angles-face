import cv2
import mediapipe as mp
import numpy as np
from math import asin, acos, degrees, sqrt

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)


def getAngles():
    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:

        while True:
            ret, frame = cap.read()
            if ret is False:
                break
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
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

                    # points
                    front = np.array([frontX, frontY])
                    chin = np.array([chinX, chinY])
                    earLeft = np.array([earLeftX, earLeftY])
                    earRight = np.array([earRightX, earRightY])
                    center = np.array([frontX, earRightY])

                    # axial geometry lines
                    s1 = np.linalg.norm(front - earLeft)
                    s2 = np.linalg.norm(front - earRight)
                    u1 = np.linalg.norm(chin - earLeft)
                    u2 = np.linalg.norm(chin - earRight)
                    h1 = np.linalg.norm(front - center)
                    w1 = np.linalg.norm(earLeft - center)
                    h2 = np.linalg.norm(chin - center)
                    w2 = np.linalg.norm(earLeft - chin)
                    e1 = np.linalg.norm(earRight - chin)

                    # lines distance
                    frontChin = np.linalg.norm(front - chin)
                    ears = np.linalg.norm(earLeft - earRight)



                    # *-*-*-*-*-* angles *-*-*-*-*-*
                    # earLeftUp = int(degrees(acos(baseLineEars / h)))
                    # frontLeft = int(degrees(acos(baseLineFront / h)))
                    # resultAngles = frontLeft + earLeftUp + 90

                    # *-*-* earLeft *-*-*
                    # earLeftUp = int(degrees(acos(w1 / s1)))
                    # earLeftDown = int(degrees(acos(w1 / u1)))
                    # angleEarLeft = earLeftUp + earLeftDown
                    # # *-*-* chin *-*-*
                    # chinLeft = int(degrees(acos(h2 / w2)))
                    # chinRight = int(degrees(acos(h2 / e1)))
                    # angleChin = chinRight + chinLeft
                    # # *-*-* front *-*-*
                    # frontLeft = int(degrees(acos(h1 / s1)))
                    # frontRight = int(degrees(acos(h1 / s2)))
                    # angleFront = frontLeft + frontRight
                    # # *-*-* angleEquilibrate *-*-*
                    # angleEquilibrate = chinLeft + frontLeft + angleEarLeft

                    # *-*-*-*-* draws *-*-*-*-*
                    # *-*-* front *-*-*
                    cv2.circle(frame, (frontX, frontY), 2, (255, 0, 255), 2)
                    cv2.circle(frame, (chinX, chinY), 2, (255, 0, 255), 2)
                    cv2.line(frame, (frontX, frontY), (chinX, chinY), (211, 0, 148), 2)
                    # *-*-*-*-*-*
                    # *-*-* ears *-*-*
                    cv2.circle(frame, (earLeftX, earLeftY), 2, (255, 0, 255), 2)
                    cv2.circle(frame, (earRightX, earRightY), 2, (255, 0, 255), 2)
                    cv2.line(frame, (earLeftX, earLeftY), (earRightX, earRightY), (211, 0, 148), 2)

                    # compute
                    cv2.putText(frame, 'front: ' + str(frontChin), (20, 20), 1, 1, (0, 255, 0), 2)
                    cv2.putText(frame, 'front: ' + str(ears), (20, 40), 1, 1, (0, 255, 0), 2)


                    # *-*-*-*-*-*
                    # angle Ear Left
                    # cv2.putText(frame, str(angleEarLeft), (earLeftX, earLeftY), 1, 1, (0, 255, 0), 2)
                    # # angle chin
                    # cv2.putText(frame, str(angleChin), (chinX, chinY), 1, 1, (0, 255, 0), 2)
                    # # angle front
                    # cv2.putText(frame, str(angleFront), (frontX, frontY), 1, 1, (0, 255, 0), 2)
                    # # center
                    # cv2.circle(frame, (frontX, earRightY), 2, (0, 0, 255), 3)
                    # cv2.putText(frame, str(angleEquilibrate), (20, 20), 1, 1, (0, 255, 0), 2)

            # shows
            cv2.imshow("Frame", frame)

            # finished capture
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

    # release capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    getAngles()
